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

# ────────────────────────────────────────────────────────────────────
# Force 'spawn' start method before any CUDA context is created
mp.set_start_method('spawn', force=True)
# ────────────────────────────────────────────────────────────────────

def parse_args():
    """
    Parse configuration and command-line arguments using OmegaConf.
    Adds support for threads_per_rank, num_workers, and pin_memory as CLI args.
    """
    # Load the YAML config
    conf = OmegaConf.load(sys.argv[1])
    # Remove the config filename from sys.argv so CLI merges start at sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()

    # Default threads_per_rank
    if "threads_per_rank" not in conf:
        conf.threads_per_rank = 4

    # Default dataloader params (override via CLI or YAML)
    if "num_workers" not in conf.train:
        conf.train.num_workers = 4
    if "pin_memory" not in conf.train:
        conf.train.pin_memory = False

    # Now freeze the config tree so no extra keys get added later
    OmegaConf.set_struct(conf, True)
    return conf


def init_wandb(config):
    rank = get_rank() if is_initialized() else 0
    if rank == 0:
        wandb.init(
            project="your_project_name",
            config=OmegaConf.to_container(config, resolve=True)
        )
        print(f"wandb initialized on rank {rank}")
    else:
        print(f"wandb skipped on rank {rank}")


def set_thread_limits(num_threads: int):
    if num_threads == -1:
        print("No thread limits set. Using system defaults.")
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

    print(f"Thread limits set to {num_threads} for all major libraries.")


def train(rank, world_size, config):
    # 1) Limit threads per process
    set_thread_limits(config.threads_per_rank)

    # 2) Initialize DDP
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Initializing Rank {rank}.")

    init_wandb(config)

    # Only rank 0 manages logging directories
    if rank == 0:
        os.makedirs(config.train.log_path, exist_ok=True)
        for sub in ("train_images", "val_images"):
            os.makedirs(os.path.join(config.train.log_path, sub), exist_ok=True)
        OmegaConf.save(config, os.path.join(config.train.log_path, "config.yaml"))

    dist.barrier()

    # 3) Build and wrap model
    trainer = CARAAffectTrainer(config).to(rank)
    trainer.device = rank
    trainer.tater.device = rank
    print(f"Rank {rank}: Model params = {sum(p.numel() for p in trainer.parameters())}")

    if config.resume:
        trainer.load_model(
            config.resume,
            load_fuse_generator=config.load_fuse_generator,
            load_encoder=config.load_encoder,
            device=f"cuda:{rank}"
        )
    trainer.create_base_encoder()
    dist.barrier()
    trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[rank])

    # 4) Initialize dataloaders with explicit num_workers & pin_memory
    train_loader, val_loader, effective_seg_count = load_dataloaders_parallel(
        config, rank, world_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory
    )

    # ---------------- Training loop ---------------- #
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        print(f"Rank {rank} starting epoch {epoch}")
        train_loader.sampler.set_epoch(epoch)
        trainer.module.configure_optimizers(effective_seg_count, epoch != 0)

        for phase in ("val", "train"):
            loader = val_loader if phase == "val" else train_loader

            if phase == "val":
                all_affect_out = []
                all_affect_gt = []
                all_video_IDs = []

            for batch_idx, batch in tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f"Rank {rank} [{phase}]",
                position=rank
            ):
                try:
                    if not batch:
                        continue

                    # quick debug: first 100 batches only
                    if batch_idx >= 100:
                        break

                    # sync all ranks before each batch
                    dist.barrier()

                    # move tensors to correct GPU
                    for k, v in batch.items():
                        if k not in ("audio_phonemes", "text_phonemes") and torch.is_tensor(v[0]):
                            batch[k] = [x.to(rank) for x in v]

                    out = trainer.module.step(batch, batch_idx, epoch, phase=phase)

                    if rank == 0 and batch_idx % config.train.visualize_every == 0:
                        wandb.log({"epoch": epoch, "batch_idx": batch_idx})

                    if phase == "val":
                        all_affect_out.append(out["valence_arousal_out"].detach().cpu().numpy())
                        all_affect_gt.append(out["valence_arousal_gt"].detach().cpu().numpy())
                        all_video_IDs.append(batch["video_ID"])

                except Exception:
                    if rank == 0:
                        print(f"Error at batch {batch_idx}, dumping traceback:")
                        traceback.print_exc()

                # save intermediate checkpoints on train phase
                if phase == "train" and rank == 0 and (
                    batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1
                ):
                    ckpt = os.path.join(config.train.log_path, f"model_{epoch}_{batch_idx}.pt")
                    trainer.module.save_model(trainer.module.state_dict(), ckpt)

            # ----------- VAL PHASE: gather + compute metrics ----------- #
            if phase == "val":
                dist.barrier()
                local_data = [all_affect_out, all_affect_gt, all_video_IDs]
                gathered = [None] * world_size if rank == 0 else None
                dist.gather_object(local_data, gathered, dst=0)

                if rank == 0:
                    # flatten & concatenate
                    outs, gts, vids = [], [], []
                    for rd_out, rd_gt, rd_ids in gathered:
                        outs.extend(rd_out)
                        gts.extend(rd_gt)
                        for x in rd_ids:
                            vids.extend(x.tolist() if isinstance(x, np.ndarray) else list(x))

                    arr_out = np.concatenate(outs, axis=0)
                    arr_gt  = np.concatenate(gts,  axis=0)
                    arr_vid = np.array(vids)

                    metrics_out = []
                    dims = arr_out.shape[-1]
                    for i in range(dims):
                        # global MSE & Pearson
                        mse = float(np.mean((arr_out[:,i] - arr_gt[:,i])**2))
                        r_all = float(metrics.pearson_r(arr_out[:,i], arr_gt[:,i]))
                        p_all = float(metrics.pearson_p(arr_out[:,i], arr_gt[:,i]))
                        metrics_out += [mse, r_all, p_all]

                        # within-video Pearson
                        vids_unique = np.unique(arr_vid)
                        r_within, p_within, nan_c = 0.0, 0.0, 0
                        for v in vids_unique:
                            mask = (arr_vid == v)
                            r_v = metrics.pearson_r(arr_out[mask,i], arr_gt[mask,i])
                            p_v = metrics.pearson_p(arr_out[mask,i], arr_gt[mask,i])
                            if np.isnan(r_v):
                                nan_c += 1
                            else:
                                r_within += r_v
                                p_within += p_v
                        valid = len(vids_unique) - nan_c
                        if valid > 0:
                            r_within /= valid
                            p_within /= valid
                        metrics_out += [r_within, p_within]

                        # between-video Pearson
                        avg_o = np.array([arr_out[arr_vid==v,i].mean() for v in vids_unique])
                        avg_g = np.array([arr_gt[arr_vid==v,i].mean() for v in vids_unique])
                        metrics_out += [
                            float(metrics.pearson_r(avg_o, avg_g)),
                            float(metrics.pearson_p(avg_o, avg_g)),
                            nan_c
                        ]

                    save_path = os.path.join(
                        config.train.log_path, f"metrics_{epoch}_{batch_idx}.npy"
                    )
                    np.save(save_path, np.array(metrics_out, dtype=float))

        # end of epoch

    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    config = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size)