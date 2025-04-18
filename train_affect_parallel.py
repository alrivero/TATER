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

# If you see CUDA-in-fork hangs, uncomment the next two lines:
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    conf.merge_with_cli()
    if "threads_per_rank" not in conf:
        conf.threads_per_rank = 4
    return conf

def init_wandb(config):
    rank = get_rank() if is_initialized() else 0
    if rank == 0:
        wandb.init(
            project="your_project_name",
            config=OmegaConf.to_container(config, resolve=True)
        )

def set_thread_limits(num_threads: int):
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
    # 1) limit threads
    set_thread_limits(config.threads_per_rank)

    # 2) initialize DDP
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    init_wandb(config)

    # 3) prepare dirs on rank 0
    if rank == 0:
        os.makedirs(config.train.log_path, exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'train_images'), exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'val_images'), exist_ok=True)
        # directory for shard files
        os.makedirs(os.path.join(config.train.log_path, 'val_shards'), exist_ok=True)
        OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    # 4) sync before model
    dist.barrier()

    # 5) build model & DDP wrap
    trainer = CARAAffectTrainer(config).to(rank)
    trainer.device = rank
    trainer.tater.device = rank
    if config.resume:
        trainer.load_model(
            config.resume,
            load_fuse_generator=config.load_fuse_generator,
            load_encoder=config.load_encoder,
            device=f'cuda:{rank}'
        )
    trainer.create_base_encoder()
    dist.barrier()
    trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[rank])

    # 6) data loaders
    train_loader, val_loader, effective_seg_count = load_dataloaders_parallel(
        config, rank, world_size
    )

    # ---------------- Training & Validation ----------------
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        train_loader.sampler.set_epoch(epoch)
        trainer.module.configure_optimizers(effective_seg_count, epoch != 0)

        for phase in ('val', 'train'):
            loader = val_loader if phase == 'val' else train_loader

            if phase == 'val':
                all_out, all_gt, all_vids = [], [], []

            for batch_idx, batch in enumerate(tqdm(
                loader,
                desc=f"Rank {rank} {phase}",
                position=rank,
                leave=False
            )):
                # **preserve this barrier** in the batch loop
                dist.barrier()

                # move tensors to GPU
                for k, v in batch.items():
                    if k in ('audio_phonemes', 'text_phonemes'):
                        continue
                    if isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                        batch[k] = [x.to(rank) for x in v]
                    elif torch.is_tensor(v):
                        batch[k] = v.to(rank)

                out = trainer.module.step(batch, batch_idx, epoch, phase=phase)

                if rank == 0 and phase == 'train' and batch_idx % config.train.visualize_every == 0:
                    wandb.log({'epoch': epoch, 'batch_idx': batch_idx})

                if phase == 'val':
                    all_out.append(out['valence_arousal_out'].detach().cpu().numpy())
                    all_gt.append(out['valence_arousal_gt'].detach().cpu().numpy())
                    all_vids.extend(batch['video_ID'])

                if phase == 'train' and rank == 0 and (
                    batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1
                ):
                    ckpt = os.path.join(
                        config.train.log_path, f'model_{epoch}_{batch_idx}.pt'
                    )
                    trainer.module.save_model(trainer.module.state_dict(), ckpt)

            # end batch loop

            if phase == 'val':
                # ---- NEW: shard to disk + gather via filesystem ----

                # 1) turn lists into numpy arrays
                local_out = np.concatenate(all_out, axis=0)      # shape [N_i, D]
                local_gt  = np.concatenate(all_gt,  axis=0)
                local_vid = np.array(all_vids, dtype=np.int64)   # shape [N_i]

                print("Starting")

                # 2) write out this rank's shard
                shard_dir = os.path.join(config.train.log_path, "val_shards")
                os.makedirs(shard_dir, exist_ok=True)
                shard_path = os.path.join(shard_dir, f"epoch{epoch}_rank{rank}.npz")
                np.savez(shard_path, out=local_out, gt=local_gt, vid=local_vid)

                print("HERE")

                # 3) sync so all shards are written
                dist.barrier()

                # 4) rank 0 reads & processes all shards
                if rank == 0:
                    files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".npz"))
                    outs_list, gts_list, vids_list = [], [], []
                    for fname in files:
                        data = np.load(os.path.join(shard_dir, fname))
                        outs_list.append(data["out"])
                        gts_list.append(data["gt"])
                        vids_list.append(data["vid"])

                    print("sorted")

                    arr_out = np.concatenate(outs_list, axis=0)
                    arr_gt  = np.concatenate(gts_list,  axis=0)
                    arr_vid = np.concatenate(vids_list, axis=0)

                    # compute metrics as before
                    n_dims = arr_out.shape[-1]
                    metrics_out = []

                    for i in range(n_dims):
                        mse   = float(np.mean((arr_out[:,i] - arr_gt[:,i])**2))
                        r_all = float(metrics.pearson_r(arr_out[:,i], arr_gt[:,i]))
                        p_all = float(metrics.pearson_p(arr_out[:,i], arr_gt[:,i]))
                        metrics_out += [mse, r_all, p_all]

                        vids_u = np.unique(arr_vid)
                        r_w, p_w, nan_c = 0.0, 0.0, 0
                        for vid in tqdm(vids_u, desc=f"Dim {i} within-video", leave=False):
                            mask = (arr_vid == vid)
                            rv = metrics.pearson_r(arr_out[mask,i], arr_gt[mask,i])
                            pv = metrics.pearson_p(arr_out[mask,i], arr_gt[mask,i])
                            if np.isnan(rv):
                                nan_c += 1
                            else:
                                r_w += rv; p_w += pv
                        valid = len(vids_u) - nan_c
                        if valid > 0:
                            r_w /= valid; p_w /= valid
                        metrics_out += [r_w, p_w]

                        avg_o = np.array([arr_out[arr_vid==v,i].mean() for v in vids_u])
                        avg_g = np.array([arr_gt[arr_vid==v,i].mean() for v in vids_u])
                        r_b = float(metrics.pearson_r(avg_o, avg_g))
                        p_b = float(metrics.pearson_p(avg_o, avg_g))
                        metrics_out += [r_b, p_b, nan_c]

                    metrics_arr = np.array(metrics_out, dtype=float)
                    print(f"[Rank 0] Epoch {epoch} metrics: {metrics_arr}")
                    np.save(
                        os.path.join(config.train.log_path, f'metrics_{epoch}.npy'),
                        metrics_arr
                    )

                    # 5) clean up shard files
                    for fname in files:
                        os.remove(os.path.join(shard_dir, fname))

        # end phase loop

    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == '__main__':
    config = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size)