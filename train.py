import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.tater_trainer import TATERTrainer
import os
from datasets.data_utils import load_dataloaders
import debug
import traceback
import warnings
import wandb
from torch.distributed import get_rank, is_initialized

def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

def batch_split_into_windows(batch, config):
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
        
        framewise_keys = ["flag_landmarks_fan", "img", "img_mica", "landmarks_fan", "landmarks_mp", "mask"]
        for key in framewise_keys:
            split_batch[key].append(batch[key][split_idx][frame_idx:frame_idx + remaining + overlap])

        batchwise_keys = ["text", "fps", "audio_sample_rate", "silent", "text_phonemes"]
        for key in batchwise_keys:
            split_batch[key].append(batch[key][split_idx])
        
        a_start = int((batch["audio_sample_rate"][split_idx] / batch["fps"][split_idx]) * frame_idx)
        a_end = int((batch["audio_sample_rate"][split_idx] / batch["fps"][split_idx]) * (frame_idx + remaining + overlap))
        split_batch["audio"].append(batch["audio"][split_idx][a_start:a_end])

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


if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    warnings.filterwarnings("ignore", message="GaussNoise could work incorrectly in ReplayMode for other input data")

    # ----------------------- initialize log directories ----------------------- #
    os.makedirs(config.train.log_path, exist_ok=True)
    train_images_save_path = os.path.join(config.train.log_path, 'train_images')
    os.makedirs(train_images_save_path, exist_ok=True)
    val_images_save_path = os.path.join(config.train.log_path, 'val_images')
    os.makedirs(val_images_save_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    train_loader, val_loader, effective_seg_count = load_dataloaders(config)

    trainer = TATERTrainer(config)
    trainer = trainer.to(config.device)

    if config.resume:
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # after loading, copy the base encoder 
    # this is used for regularization w.r.t. the base model as well as to compare the results    
    trainer.create_base_encoder()
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):        
        # restart everything at each epoch!
        trainer.configure_optimizers(effective_seg_count, epoch != 0)

        for phase in ['train']:
            loader = train_loader if phase == 'train' else val_loader
            
            batch_idx = 0
            for segment_idx, batch in tqdm(enumerate(loader), total=len(loader)):
                try:
                    if not batch:
                        continue

                    split_batches = batch_split_into_windows(batch, config)
                    all_outputs = []
                    for test_idx, split_batch in enumerate(split_batches):
                        # Move batch to the appropriate device
                        trainer.set_freeze_status(config, batch_idx, epoch)
                        for key in split_batch:
                            extra_keys = ["audio_phonemes", "text_phonemes", "og_idxs", "removed_frames", "roberta", "roberta_pca_5", "valence_arousal"]
                            if key not in extra_keys and torch.is_tensor(split_batch[key][0]):
                                for i in range(len(split_batch[key])):
                                    split_batch[key][i] = split_batch[key][i].to(config.device)

                        # Perform training/validation step
                        all_outputs.append(trainer.step(split_batch, batch_idx,epoch, phase=phase))

                        for key in split_batch:
                            extra_keys = ["audio_phonemes", "text_phonemes", "og_idxs", "removed_frames", "roberta", "roberta_pca_5", "valence_arousal"]
                            if key not in extra_keys and torch.is_tensor(split_batch[key][0]):
                                for i in range(len(split_batch[key])):
                                    split_batch[key][i] = split_batch[key][i].to("cpu")
                        batch_idx += 1

                    if segment_idx % config.train.visualize_every == 0:
                        all_vis = []
                        for (split_batch, outputs) in zip(split_batches, all_outputs):
                            all_vis.append(trainer.create_visualizations(split_batch, outputs))

                        trainer.save_visualizations(
                            all_vis,
                            f"{config.train.log_path}/{phase}_images/{epoch}_{segment_idx}.jpg",
                            f"{config.train.log_path}/{phase}_images/{epoch}_{segment_idx}.mp4",
                            frame_overlap=config.train.split_overlap,
                            show_landmarks=True
                        )

                except Exception as e:
                    print(f"Error loading batch_idx {batch_idx}!")
                    print(e)
                    traceback.print_exc()

                if batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1:
                    trainer.save_model(trainer.state_dict(), os.path.join(config.train.log_path, f'model_{epoch}_{batch_idx}.pt'))
