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
    print("parse_args: loading config from", sys.argv[1])
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    if "threads_per_rank" not in conf:
        conf.threads_per_rank = 4
    print("parse_args: threads_per_rank =", conf.threads_per_rank)
    return conf

def init_wandb(config):
    rank = get_rank() if is_initialized() else 0
    if rank == 0:
        print("init_wandb: initializing on rank 0")
        wandb.init(project="your_project_name", config=OmegaConf.to_container(config, resolve=True))
    else:
        print(f"init_wandb: skipping on rank {rank}")

def set_thread_limits(num_threads: int):
    print(f"set_thread_limits: setting to {num_threads}")
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
    print(f"[Rank {rank}] train(): start")
    set_thread_limits(config.threads_per_rank)

    print(f"[Rank {rank}] init_process_group")
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    init_wandb(config)

    if rank == 0:
        print("[Rank 0] creating log dirs")
        os.makedirs(config.train.log_path, exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'train_images'), exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'val_images'), exist_ok=True)
        OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    print(f"[Rank {rank}] barrier before model init")
    dist.barrier()

    print(f"[Rank {rank}] building model")
    trainer = CARAAffectTrainer(config).to(rank)
    trainer.device = rank
    trainer.tater.device = rank
    print(f"[Rank {rank}] num_params = {sum(p.numel() for p in trainer.parameters())}")

    if config.resume:
        print(f"[Rank {rank}] loading checkpoint {config.resume}")
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
    print(f"[Rank {rank}] wrapped in DDP")

    print(f"[Rank {rank}] loading dataloaders")
    train_loader, val_loader, effective_seg_count = load_dataloaders_parallel(
        config, rank, world_size
    )
    print(f"[Rank {rank}] train_loader length = {len(train_loader)}, val_loader length = {len(val_loader)}")

    # ---------------- Training loop ---------------- #
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        print(f"[Rank {rank}] === Epoch {epoch}/{config.train.num_epochs} ===")
        train_loader.sampler.set_epoch(epoch)
        trainer.module.configure_optimizers(effective_seg_count, epoch != 0)

        for phase in ['val', 'train']:
            print(f"[Rank {rank}] Starting phase '{phase}'")
            loader = train_loader if phase == 'train' else val_loader

            if phase == 'val':
                all_affect_out = []
                all_affect_gt = []
                all_video_IDs = []

            for batch_idx, batch in enumerate(loader):
                dist.barrier()
                if not batch:
                    continue
                print(f"[Rank {rank}] phase={phase} batch_idx={batch_idx}")
                # move tensors to GPU
                for k, v in batch.items():
                    if k not in ("audio_phonemes", "text_phonemes") and torch.is_tensor(v[0]):
                        batch[k] = [x.to(rank) for x in v]

                out = trainer.module.step(batch, batch_idx, epoch, phase=phase)
                if rank == 0 and phase == 'train' and batch_idx % config.train.visualize_every == 0:
                    wandb.log({"epoch": epoch, "batch_idx": batch_idx})

                if phase == 'val':
                    all_affect_out.append(out["valence_arousal_out"].detach().cpu().numpy())
                    all_affect_gt.append(out["valence_arousal_gt"].detach().cpu().numpy())
                    all_video_IDs.append(batch["video_ID"])

                if phase == 'train' and rank == 0 and (
                    batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1
                ):
                    ckpt_path = os.path.join(config.train.log_path, f'model_{epoch}_{batch_idx}.pt')
                    trainer.module.save_model(trainer.module.state_dict(), ckpt_path)
                    print(f"[Rank 0] saved checkpoint {ckpt_path}")

            print(f"[Rank {rank}] Completed batch loop for phase '{phase}'")

            if phase == 'val':
                print(f"[Rank {rank}] barrier before gather")
                dist.barrier()
                print(f"[Rank {rank}] gathering data")
                local_data = [all_affect_out, all_affect_gt, all_video_IDs]
                gathered = [None] * world_size if rank == 0 else None
                dist.gather_object(local_data, gathered, dst=0)
                print(f"[Rank {rank}] gather_object returned")

                if rank == 0:
                    print(f"[Rank 0] Received data from all ranks, computing metrics")
                    merged_out, merged_gt, merged_ids = [], [], []
                    for rd_out, rd_gt, rd_ids in gathered:
                        merged_out.extend(rd_out)
                        merged_gt.extend(rd_gt)
                        for arr in rd_ids:
                            merged_ids.extend(arr.tolist() if isinstance(arr, np.ndarray) else list(arr))

                    all_affect_out = np.concatenate(merged_out, axis=0)
                    all_affect_gt  = np.concatenate(merged_gt,  axis=0)
                    all_video_IDs  = np.array(merged_ids)

                    n_dims = all_affect_out.shape[-1]
                    metrics_out = []
                    for i in range(n_dims):
                        # global
                        mse = float(np.mean((all_affect_out[:,i] - all_affect_gt[:,i])**2))
                        r_all = float(metrics.pearson_r(all_affect_out[:,i], all_affect_gt[:,i]))
                        p_all = float(metrics.pearson_p(all_affect_out[:,i], all_affect_gt[:,i]))

                        video_ids = np.unique(all_video_IDs)
                        r_within, p_within, nan_c = 0.0, 0.0, 0
                        for vid in tqdm(video_ids, desc=f"Dim {i} within-video", leave=False):
                            mask = (all_video_IDs == vid)
                            r_vid = metrics.pearson_r(all_affect_out[mask,i], all_affect_gt[mask,i])
                            if np.isnan(r_vid):
                                nan_c += 1
                            else:
                                r_within += r_vid
                                p_within += metrics.pearson_p(all_affect_out[mask,i], all_affect_gt[mask,i])

                        valid = len(video_ids) - nan_c
                        if valid > 0:
                            r_within /= valid
                            p_within /= valid

                        avg_o = np.array([all_affect_out[all_video_IDs==v,i].mean() for v in video_ids])
                        avg_g = np.array([all_affect_gt[ all_video_IDs==v,i].mean() for v in video_ids])
                        r_between = float(metrics.pearson_r(avg_o, avg_g))
                        p_between = float(metrics.pearson_p(avg_o, avg_g))

                        metrics_out += [
                            mse, r_all, p_all,
                            r_within, p_within,
                            r_between, p_between,
                            nan_c
                        ]

                    metrics_arr = np.array(metrics_out, dtype=float)
                    print(f"[Rank 0] Metrics array:\n{metrics_arr}")

                    # formatted print
                    for i in range(n_dims):
                        start = i*8
                        m, ra, pa, rw, pw, rb, pb, nc = metrics_arr[start:start+8]
                        print(f"[Rank 0] Dim {i}: MSE={m:.4f}, all_r={ra:.4f}, all_p={pa:.2e}, "
                              f"within_r={rw:.4f}, within_p={pw:.2e}, between_r={rb:.4f}, between_p={pb:.2e}, nans={int(nc)}")

                    metrics_path = os.path.join(config.train.log_path, f"metrics_{epoch}_{batch_idx}.npy")
                    np.save(metrics_path, metrics_arr)
                    print(f"[Rank 0] saved metrics to {metrics_path}")

        print(f"[Rank {rank}] Finished epoch {epoch}")

    if rank == 0:
        print("[Rank 0] finishing wandb")
        wandb.finish()

    print(f"[Rank {rank}] destroying process group")
    dist.destroy_process_group()
    print(f"[Rank {rank}] train() exiting")

if __name__ == '__main__':
    print("Main: starting")
    config = parse_args()
    print("Main: config parsed")
    world_size = torch.cuda.device_count()
    print(f"Main: spawning {world_size} processes")
    mp.spawn(train, args=(world_size, config), nprocs=world_size)
    print("Main: mp.spawn returned (should only appear if processes exit cleanly)")