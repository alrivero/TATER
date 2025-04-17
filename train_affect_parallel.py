import os
import sys
import cv2
import torch
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from src.cara_affect_trainer import CARAAffectTrainer
from datasets.data_utils import load_dataloaders_parallel
from torch.distributed import get_rank, is_initialized
from src.utils import metrics
import traceback


def parse_args():
    """
    Parse configuration and command-line arguments using OmegaConf.
    Adds support for threads_per_rank as a command-line argument.
"""
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the configuration file name from sys.argv
    conf.merge_with_cli()
    
    # Default value for threads_per_rank if not provided via CLI
    if "threads_per_rank" not in conf:
        conf.threads_per_rank = 4  # Default to 4 threads per rank

    return conf

def init_wandb(config):
    rank = 0
    if is_initialized():
        rank = get_rank()

    if rank == 0:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(project="your_project_name", config=config_dict)
        print(f"wandb initialized on rank {rank}")
    else:
        print(f"wandb skipped on rank {rank}")

def set_thread_limits(num_threads: int):
    """
    Configure thread limits across major ML libraries and scientific computing frameworks.
    This ensures each rank operates with the specified thread limits.
    """
    if num_threads == -1:
        print("No thread limits set. Using system defaults.")
        return  # Do nothing, use system defaults

    os.environ["OMP_NUM_THREADS"] = str(num_threads)  # OpenMP threads
    os.environ["MKL_NUM_THREADS"] = str(num_threads)  # Intel MKL threads
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)  # OpenBLAS threads
    os.environ["NUMEXPR_MAX_THREADS"] = str(num_threads)  # NumExpr threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)  # Accelerate/VecLib threads
    os.environ["BLIS_NUM_THREADS"] = str(num_threads)  # BLIS threads
    os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={num_threads}"  # JAX/XLA threads

    # PyTorch thread limits
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    # OpenCV thread limits
    cv2.setNumThreads(num_threads)

    print(f"Thread limits set to {num_threads} for all major libraries.")

def train(rank, world_size, config):
    # Limit threads per rank
    num_threads = config.threads_per_rank
    set_thread_limits(num_threads)

    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print(f"Initializing Rank {rank}.")
    init_wandb(config)

    # Initialize paths and data loaders
    if rank == 0:
        os.makedirs(config.train.log_path, exist_ok=True)
        train_images_save_path = os.path.join(config.train.log_path, 'train_images')
        os.makedirs(train_images_save_path, exist_ok=True)
        val_images_save_path = os.path.join(config.train.log_path, 'val_images')
        os.makedirs(val_images_save_path, exist_ok=True)
        OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    # Synchronize processes before further initialization
    dist.barrier()

    # Initialize trainer and wrap in DDP
    trainer = CARAAffectTrainer(config).to(rank)
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
        train_loader.sampler.set_epoch(epoch)
        trainer.module.configure_optimizers(effective_seg_count, epoch != 0)

        for phase in ['val', 'train']:
            loader = train_loader if phase == 'train' else val_loader
            if phase == 'val':
                all_affect_out = []
                all_affect_gt = []
                all_video_IDs = []

            for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Rank {rank} Progress", position=rank):                
                try:
                    if not batch:
                        continue

                    dist.barrier()

                    for key in batch:
                        if key not in ["audio_phonemes", "text_phonemes"] and torch.is_tensor(batch[key][0]):
                            batch[key] = [x.to(rank) for x in batch[key]]

                    out = trainer.module.step(batch, batch_idx, epoch, phase=phase)
                    if rank == 0 and batch_idx % config.train.visualize_every == 0:
                        wandb.log({"epoch": epoch, "batch_idx": batch_idx})

                    if phase == 'val':
                        all_affect_out.append(out["valence_arousal"].detach().cpu().numpy())
                        all_affect_gt.append(out["valence_arousal_gt"].detach().cpu().numpy())
                        all_video_IDs.append(batch["video_ID"])

                except Exception as e:
                    if rank == 0:
                        print(f"Error loading batch_idx {batch_idx}!")
                        error_message = traceback.format_exc()
                        print(f"Error captured: {error_message}")
                        print(e)

                if phase == 'train' and rank == 0 and (batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1):
                    trainer.module.save_model(trainer.module.state_dict(), os.path.join(config.train.log_path, f'model_{epoch}_{batch_idx}.pt'))
            
            if phase == 'val':
                MSE = torch.nn.MSELoss()
                all_affect_out = np.concatenate(all_affect_out)
                all_affect_gt = np.concatenate(all_affect_gt)
                all_video_IDs = np.concatenate(all_video_IDs)
                metrics_out = []

                for i in range(all_affect_out.shape[-1]):
                    mse = MSE(all_affect_out[:, i], all_affect_gt[:, i])
                    r_all = metrics.pearson_r(all_affect_out[:, i], all_affect_gt[:, i])
                    p_all = metrics.pearson_p(all_affect_out[:, i], all_affect_gt[:, i])
                    metrics_out.append(mse)
                    metrics_out.append(r_all)
                    metrics_out.append(p_all)

                    avg_vid_out = []
                    avg_vid_gt = []
                    nan_count = 0
                    r_within = 0
                    p_within = 0
                    for j, video_ID in enumerate(all_video_IDs.unique()):
                        video_mask = all_video_IDs == video_ID
                        affect_out = all_affect_out[video_mask]
                        affect_gt = all_affect_gt[video_mask]

                        data_out = affect_out[:, i]
                        data_gt = affect_gt[:, i]

                        r_vid = metrics.pearson_r(data_out, data_gt)
                        p_vid = metrics.pearson_p(data_out, data_gt)

                        if np.isnan(r_within):
                            r_within = 0.0
                            p_within = 0.0 
                            nan_count += 1
                        else:
                            r_within += r_vid
                            p_within += p_vid

                        avg_vid_out.append(data_out.mean())
                        avg_vid_gt.append(data_gt.mean())
                    
                    r_within /= (j + 1 - nan_count)
                    p_within /= (j + 1 - nan_count)
                    metrics_out.append(r_within)
                    metrics_out.append(p_within)

                    avg_vid_out = np.array(avg_vid_out)
                    avg_vid_gt = np.array(avg_vid_gt)
                    r_between = metrics.pearson_r(avg_vid_out, avg_vid_gt)
                    p_between = metrics.pearson_p(avg_vid_out, avg_vid_gt)
                    metrics_out.append(r_between)
                    metrics_out.append(p_between)
                    metrics_out.append(nan_count)
            
                np.save(os.path.join(config.train.log_path, f'metrics_{epoch}_{batch_idx}.npy'), metrics_out)

    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == '__main__':
    config = parse_args()
    world_size = torch.cuda.device_count()

    # Launch processes for each GPU
    mp.spawn(train, args=(world_size, config), nprocs=world_size)
