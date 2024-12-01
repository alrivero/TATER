import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.tater_trainer_parallel import TATERTrainerParallel
import os
from datasets.data_utils import load_dataloaders_parallel
import traceback
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast  # For mixed precision
import wandb
from torch.distributed import get_rank, is_initialized

def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the configuration file name from sys.argv
    conf.merge_with_cli()
    return conf

def init_wandb(config):
    rank = 0
    if is_initialized():
        rank = get_rank()

    if rank == 0:
        # Convert OmegaConf to a plain dictionary
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(project="your_project_name", config=config_dict)
        print(f"wandb initialized on rank {rank}")
    else:
        print(f"wandb skipped on rank {rank}")

def batch_split_into_windows(batch, config, framewise_keys, batchwise_keys, phonemes_present=False):
    split_batches = []
    split_batch = {}
    max_b_len = config.train.max_batch_len
    overlap_lr = config.train.split_overlap
    split_idx = 0
    frame_idx = 0
    current_batch_size = 0
    remaining = max_b_len

    for key in batch.keys():
        split_batch[key] = []
    split_batch["og_idxs"] = []

    while split_idx < len(batch["img"]):
        if frame_idx == 0:
            overlap = 0
        else:
            overlap = overlap_lr
        
        for key in framewise_keys:
            split_batch[key].append(batch[key][split_idx][frame_idx:frame_idx + remaining + overlap])

        for key in batchwise_keys:
            split_batch[key].append(batch[key][split_idx])
        
        a_start = int((batch["audio_sample_rate"][split_idx] / batch["fps"][split_idx]) * frame_idx)
        a_end = int((batch["audio_sample_rate"][split_idx] / batch["fps"][split_idx]) * (frame_idx + remaining + overlap))
        split_batch["audio"].append(batch["audio"][split_idx][a_start:a_end])

        if phonemes_present:
            if len(batch["audio_phonemes"][split_idx]) != 0:
                ap_start = int((frame_idx / len(batch["img"][split_idx])) * len(batch["audio_phonemes"][split_idx][0]))
                ap_end = int(((frame_idx + remaining + overlap) / len(batch["img"][split_idx])) * len(batch["audio_phonemes"][split_idx][0]))
                split_batch["audio_phonemes"].append(batch["audio_phonemes"][split_idx][:, ap_start:ap_end])
            else:
                split_batch["audio_phonemes"].append(torch.empty((1,)))

            if len(batch["phoneme_timestamps"][split_idx]) > 0:
                adjusted_phonemes = [(x, y, s - frame_idx, e - frame_idx) for (x, y, s, e) in batch["phoneme_timestamps"][split_idx] if (s >= frame_idx and e < frame_idx + remaining + overlap)]
                split_batch["phoneme_timestamps"].append(adjusted_phonemes)
            else:
                split_batch["phoneme_timestamps"].append([])

        split_batch["og_idxs"].append((frame_idx, min(frame_idx + remaining + overlap, len(batch["img"][split_idx]))))

        if frame_idx == 0:
            frame_idx -= overlap_lr
        frame_idx = min(frame_idx + remaining, len(batch["img"][split_idx]) - overlap) % (len(batch["img"][split_idx]) - overlap)
        if frame_idx == 0:
            split_idx += 1
    
        current_batch_size += (len(split_batch["img"][-1]))
        if current_batch_size >= max_b_len:
            split_batches.append(split_batch)
            split_batch = {}
            for key in batch.keys():
                split_batch[key] = []
            split_batch["og_idxs"] = []

            remaining = max_b_len
            current_batch_size = 0
        else:
            remaining -= current_batch_size
    
    if len(split_batch["img"]) > 0:
        split_batches.append(split_batch)
    
    return split_batches

def train(rank, world_size, config):

    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print(f"Initializing Rank {rank}.")
    init_wandb(config)

    # Initialize paths and data loaders
    os.makedirs(config.train.log_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    # Synchronize processes before further initialization
    dist.barrier()

    # Initialize trainer and wrap in DDP
    trainer = TATERTrainerParallel(config).to(rank)
    trainer.device = rank
    trainer.tater.device = rank
    num_params = sum(p.numel() for p in trainer.parameters())
    print(f"Rank {rank}: Model initialized with {num_params} parameters.")

    # Load checkpoint if specified
    if config.resume:
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=f"cuda:{rank}")
    trainer.create_base_encoder()

    # Synchronize before wrapping in DDP
    dist.barrier()
    trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[rank])

    # Initialize data loaders
    train_loader, val_loader, effective_seg_count = load_dataloaders_parallel(config, rank, world_size)


    # ---------------- Training loop ---------------- #
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        print(f"Rank {rank} HERE")
        # Update the sampler's epoch to shuffle data properly
        train_loader.sampler.set_epoch(epoch)
        trainer.module.configure_optimizers(effective_seg_count, epoch != 0)

        for phase in ['train']:
            loader = train_loader if phase == 'train' else val_loader

            batch_idx = 0
            print(f"Rank {rank} initialized for Epoch {epoch}.")
            for segment_idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Rank {rank} Progress", position=rank):                
                try:
                    if not batch:
                        continue

                    framewise_keys = ["flag_landmarks_fan", "img", "img_mica", "landmarks_fan", "landmarks_mp", "mask"]
                    batchwise_keys = ["fps", "audio_sample_rate", "silent"]
                    extra_keys = [
                        "audio_phonemes",
                        "text_phonemes",
                        "og_idxs",
                        "removed_frames",
                        "roberta",
                        "roberta_pca_5",
                        "valence_arousal",
                        "hubert_frame_level_embeddings",
                        "wav2vec2_frame_level_embeddings",
                        "phoneme_timestamps",
                        "text_phonemes"
                    ]
                    to_concat = ['flag_landmarks_fan', 'img', 'img_mica', 'landmarks_fan', 'landmarks_mp', 'mask']

                    split_batches = batch_split_into_windows(batch, config, framewise_keys, batchwise_keys)
                    all_outputs = []

                    for split_batch in split_batches:
                        dist.barrier()
                        trainer.module.set_freeze_status(config, batch_idx, epoch)

                        for key in split_batch:
                            if key not in extra_keys and torch.is_tensor(split_batch[key][0]):
                                split_batch[key] = [x.to(rank) for x in split_batch[key]]

                        all_outputs.append(trainer.module.step(split_batch, batch_idx, epoch, phase=phase))

                        for key in split_batch:
                            if key not in extra_keys and torch.is_tensor(split_batch[key][0]):
                                if key in to_concat:
                                    split_batch[key] = split_batch[key].to("cpu")
                                else:
                                    split_batch[key] = [x.to("cpu") for x in split_batch[key]]

                        batch_idx += 1

                    # Save visualizations and log metrics on rank 0 only
                    if rank == 0 and segment_idx % config.train.visualize_every == 0:
                        all_vis = []
                        for (split_batch, outputs) in zip(split_batches, all_outputs):
                            all_vis.append(trainer.module.create_visualizations(split_batch, outputs))

                        trainer.module.save_visualizations(
                            all_vis,
                            f"{config.train.log_path}/{phase}_images/{epoch}_{segment_idx}.jpg",
                            f"{config.train.log_path}/{phase}_images/{epoch}_{segment_idx}.mp4",
                            frame_overlap=config.train.split_overlap,
                            show_landmarks=True,
                        )

                        # Log metrics and visualizations to wandb
                        wandb.log({"epoch": epoch, "batch_idx": batch_idx, "segment_idx": segment_idx})
                        wandb.log({"visualization": wandb.Image(f"{config.train.log_path}/{phase}_images/{epoch}_{segment_idx}.jpg")})

                except Exception as e:
                    if rank == 0:  # Print errors only on rank 0
                        print(f"Error loading batch_idx {batch_idx}!")
                        print(e)
                        traceback.print_exc()

                # Save model checkpoints only on rank 0
                if rank == 0 and (batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1):
                    trainer.module.save_model(trainer.module.state_dict(), os.path.join(config.train.log_path, f'model_{epoch}_{batch_idx}.pt'))

    # Finalize wandb only on rank 0
    if rank == 0:
        wandb.finish()

    # Clean up the process group
    dist.destroy_process_group()


if __name__ == '__main__':
    # Parse configuration
    config = parse_args()
    world_size = torch.cuda.device_count()

    # Launch processes for each GPU
    mp.spawn(train, args=(world_size, config), nprocs=world_size)
