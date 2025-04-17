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

# If you ever want spawn instead of fork (fixes some PyTorch+CUDA hiccups)
# mp.set_start_method('spawn', force=True)

def parse_args():
    print("[parse_args] Loading config from", sys.argv[1])
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    if "threads_per_rank" not in conf:
        conf.threads_per_rank = 4
    print(f"[parse_args] threads_per_rank = {conf.threads_per_rank}")
    return conf

def init_wandb(config):
    rank = get_rank() if is_initialized() else 0
    if rank == 0:
        print("[init_wandb] Initializing W&B on rank 0")
        wandb.init(
            project="your_project_name",
            config=OmegaConf.to_container(config, resolve=True)
        )
    else:
        print(f"[init_wandb] Skipping W&B on rank {rank}")

def set_thread_limits(num_threads: int):
    print(f"[set_thread_limits] Setting threads to {num_threads}")
    if num_threads == -1:
        return
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["BLIS_NUM_THREADS"] = str(num_threads)
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        f"intra_op_parallelism_threads={num_threads}"
    )
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    cv2.setNumThreads(num_threads)

def train(rank, world_size, config):
    print(f"[Rank {rank}] train() start")
    set_thread_limits(config.threads_per_rank)

    print(f"[Rank {rank}] init_process_group")
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    init_wandb(config)

    if rank == 0:
        print("[Rank 0] Creating log directories")
        os.makedirs(config.train.log_path, exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'train_images'), exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'val_images'), exist_ok=True)
        OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    print(f"[Rank {rank}] barrier before model init")
    dist.barrier()

    print(f"[Rank {rank}] Building model")
    trainer = CARAAffectTrainer(config).to(rank)
    trainer.device = rank
    trainer.tater.device = rank
    print(f"[Rank {rank}] Model params = {sum(p.numel() for p in trainer.parameters())}")
    if config.resume:
        print(f"[Rank {rank}] Loading checkpoint {config.resume}")
        trainer.load_model(
            config.resume,
            load_fuse_generator=config.load_fuse_generator,
            load_encoder=config.load_encoder,
            device=f"cuda:{rank}"
        )
    trainer.create_base_encoder()

    print(f"[Rank {rank}] barrier before DDP wrap")
    dist.barrier()
    trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[rank])
    print(f"[Rank {rank}] Wrapped model in DDP")

    print(f"[Rank {rank}] Loading dataloaders")
    train_loader, val_loader, effective_seg_count = load_dataloaders_parallel(
        config, rank, world_size
    )
    print(f"[Rank {rank}] train_loader={len(train_loader)}, val_loader={len(val_loader)}")

    # ---------------- Training loop ---------------- #
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        print(f"[Rank {rank}] === Epoch {epoch}/{config.train.num_epochs} ===")
        train_loader.sampler.set_epoch(epoch)
        trainer.module.configure_optimizers(effective_seg_count, epoch != 0)

        for phase in ['val', 'train']:
            print(f"[Rank {rank}] Starting phase '{phase}'")
            loader = train_loader if phase == 'train' else val_loader

            if phase == 'val':
                all_affect_out, all_affect_gt, all_video_IDs = [], [], []

            for batch_idx, batch in tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f"Rank {rank} {phase}",
                position=rank
            ):
                # no per-batch barrier—ensures barrier counts match
                try:
                    # ensure all ranks sync each batch
                    dist.barrier()

                    # move tensors to GPU
                    for k, v in batch.items():
                        if k in ("audio_phonemes", "text_phonemes"):
                            continue
                        if isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                            batch[k] = [x.to(rank) for x in v]
                        elif torch.is_tensor(v):
                            batch[k] = v.to(rank)

                    out = trainer.module.step(batch, batch_idx, epoch, phase=phase)

                    if rank == 0 and phase == 'train' and batch_idx % config.train.visualize_every == 0:
                        wandb.log({"epoch": epoch, "batch_idx": batch_idx})

                    if phase == 'val':
                        all_affect_out.append(out["valence_arousal_out"].detach().cpu().numpy())
                        all_affect_gt.append(out["valence_arousal_gt"].detach().cpu().numpy())
                        all_video_IDs.append(batch["video_ID"])

                except Exception:
                    if rank == 0:
                        print(f"[Rank {rank}] Error in batch {batch_idx}")
                        traceback.print_exc()

                # save checkpoints in train phase
                if phase == 'train' and rank == 0 and (
                    batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1
                ):
                    ckpt = os.path.join(config.train.log_path, f"model_{epoch}_{batch_idx}.pt")
                    trainer.module.save_model(trainer.module.state_dict(), ckpt)
                    print(f"[Rank 0] Saved checkpoint {ckpt}")

            print(f"[Rank {rank}] Completed batch loop for phase '{phase}'")

            if phase == 'val':
                print(f"[Rank {rank}] barrier before gather")
                dist.barrier()   # sync all ranks here
                print(f"[Rank {rank}] Gathering from all ranks")
                local_data = [all_affect_out, all_affect_gt, all_video_IDs]
                gathered = [None for _ in range(world_size)] if rank == 0 else None
                dist.gather_object(local_data, gathered, dst=0)
                print(f"[Rank {rank}] gather_object done")

                if rank == 0:
                    print("[Rank 0] Received data, start metric computation")
                    merged_out, merged_gt, merged_ids = [], [], []
                    for rd_out, rd_gt, rd_ids in gathered:
                        merged_out.extend(rd_out)
                        merged_gt.extend(rd_gt)
                        for arr in rd_ids:
                            merged_ids.extend(arr.tolist() if isinstance(arr, np.ndarray) else list(arr))

                    arr_out = np.concatenate(merged_out, axis=0)
                    arr_gt  = np.concatenate(merged_gt, axis=0)
                    arr_vid = np.array(merged_ids)
                    n_dims = arr_out.shape[-1]

                    print(f"[Rank 0] Computing metrics on {arr_out.shape[0]} samples across {len(np.unique(arr_vid))} videos")
                    metrics_out = []
                    for i in range(n_dims):
                        print(f"[Rank 0] --> Dimension {i}")
                        mse_val = float(np.mean((arr_out[:,i] - arr_gt[:,i])**2))
                        r_all = float(metrics.pearson_r(arr_out[:,i], arr_gt[:,i]))
                        p_all = float(metrics.pearson_p(arr_out[:,i], arr_gt[:,i]))
                        metrics_out += [mse_val, r_all, p_all]

                        vids = np.unique(arr_vid)
                        print(f"[Rank 0] ... within-video loop for dim {i} ({len(vids)} vids)")
                        r_within, p_within, nan_c = 0.0, 0.0, 0
                        for vid in tqdm(vids, desc=f"Dim {i} within-video", leave=False):
                            mask = (arr_vid == vid)
                            rv = metrics.pearson_r(arr_out[mask,i], arr_gt[mask,i])
                            pv = metrics.pearson_p(arr_out[mask,i], arr_gt[mask,i])
                            if np.isnan(rv):
                                nan_c += 1
                            else:
                                r_within += rv
                                p_within += pv
                        valid = len(vids) - nan_c
                        if valid > 0:
                            r_within /= valid
                            p_within /= valid
                        metrics_out += [r_within, p_within]

                        print(f"[Rank 0] ... between-video for dim {i}")
                        avg_o = np.array([arr_out[arr_vid==v,i].mean() for v in vids])
                        avg_g = np.array([arr_gt[arr_vid==v,i].mean() for v in vids])
                        r_between = float(metrics.pearson_r(avg_o, avg_g))
                        p_between = float(metrics.pearson_p(avg_o, avg_g))
                        metrics_out += [r_between, p_between, nan_c]

                    metrics_arr = np.array(metrics_out, dtype=float)
                    print(f"[Rank 0] Final metrics array:\n{metrics_arr}")

                    save_path = os.path.join(config.train.log_path, f"metrics_{epoch}_{batch_idx}.npy")
                    np.save(save_path, metrics_arr)
                    print(f"[Rank 0] Saved metrics to {save_path}")

        print(f"[Rank {rank}] Finished epoch {epoch}")
        dist.barrier()

    if rank == 0:
        print("[Rank 0] Finishing wandb")
        wandb.finish()

    print(f"[Rank {rank}] Destroying process group")
    dist.destroy_process_group()
    print(f"[Rank {rank}] train() exit")

if __name__ == '__main__':
    print("[Main] Starting")
    config = parse_args()
    world_size = torch.cuda.device_count()
    print(f"[Main] Spawning {world_size} processes")
    mp.spawn(train, args=(world_size, config), nprocs=world_size)
    print("[Main] Spawn exited")