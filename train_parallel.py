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
from src.tater_trainer_parallel import TATERTrainerParallel
from datasets.data_utils import load_dataloaders_parallel
from torch.distributed import get_rank, is_initialized
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
        train_loader.sampler.set_epoch(epoch)
        trainer.module.configure_optimizers(effective_seg_count, epoch != 0)

        for phase in ['train']:
            loader = train_loader if phase == 'train' else val_loader
            for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Rank {rank} Progress", position=rank):                
                try:
                    if not batch:
                        continue

                    dist.barrier()
                    trainer.module.set_freeze_status(config, batch_idx, epoch)

                    for key in batch:
                        if key not in ["audio_phonemes", "text_phonemes"] and torch.is_tensor(batch[key][0]):
                            batch[key] = [x.to(rank) for x in batch[key]]

                    output = trainer.module.step(batch, batch_idx, epoch, phase=phase)

                    if rank == 0 and batch_idx % config.train.visualize_every == 0:
                        all_vis = [trainer.module.create_visualizations(batch, output)]
                        trainer.module.save_visualizations(
                            all_vis,
                            f"{config.train.log_path}/{phase}_images/{epoch}_{batch_idx}",
                            f"{config.train.log_path}/{phase}_images/{epoch}_{batch_idx}.mp4",
                            frame_overlap=config.train.split_overlap,
                            show_landmarks=True,
                        )

                        wandb.log({"epoch": epoch, "batch_idx": batch_idx})

                except Exception as e:
                    if rank == 0:
                        print(f"Error loading batch_idx {batch_idx}!")
                        error_message = traceback.format_exc()
                        print(f"Error captured: {error_message}")
                        print(e)
                        exit()

                if rank == 0 and (batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1):
                    trainer.module.save_model(trainer.module.state_dict(), os.path.join(config.train.log_path, f'model_{epoch}_{batch_idx}.pt'))

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    cfg = parse_args()

    # ───── ONE‑GPU DEBUG MODE ─────────────────────────────────────
    # export DEBUG_ONE=1 before running → everything happens in‑process
    if os.getenv("DEBUG_ONE") == "1":          # ❶ add this
        train(rank=0, world_size=1, config=cfg)  # ❷ and this
        sys.exit(0)
    # ──────────────────────────────────────────────────────────────

    world_size = torch.cuda.device_count()
    mp.spawn(train, nprocs=world_size, args=(world_size, cfg))
