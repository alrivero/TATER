import os
import sys
import cv2
import torch
import wandb
import pickle
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
import math

# If you see CUDA‐in‐fork hangs, uncomment:
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import warnings

# suppress exactly that GaussNoise-in-ReplayCompose warning
warnings.filterwarnings(
    "ignore",
    message=".*GaussNoise could work incorrectly in ReplayMode for other input data because its' params depend on targets.*",
    category=UserWarning,
    module=r"albumentations\.core\.transforms_interface"
)

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
            project="CARA_Affect",
            config=OmegaConf.to_container(config, resolve=True)
        )

def set_thread_limits(num_threads: int):
    if num_threads == -1:
        return
    os.environ["OMP_NUM_THREADS"]     = str(num_threads)
    os.environ["MKL_NUM_THREADS"]     = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"]= str(num_threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["BLIS_NUM_THREADS"]    = str(num_threads)
    os.environ["XLA_FLAGS"]           = (
        "--xla_cpu_multi_thread_eigen=false "
        f"intra_op_parallelism_threads={num_threads}"
    )
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    cv2.setNumThreads(num_threads)

def train(rank, world_size, config):
    # 1) limit threads
    set_thread_limits(config.threads_per_rank)

    # 2) init DDP
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    init_wandb(config)

    # 3) prepare dirs on rank 0
    if rank == 0:
        os.makedirs(config.train.log_path, exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'train_images'), exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'val_images'), exist_ok=True)
        os.makedirs(os.path.join(config.train.log_path, 'val_shards'), exist_ok=True)
        OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    dist.barrier()

    # 4) build model & wrap DDP
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

    # 5) data loaders
    train_loader, val_loader, _ = load_dataloaders_parallel(config, rank, world_size)

    # 6) configure optimizer+scheduler once for all epochs
    batches_per_epoch = len(train_loader)
    steps_per_epoch  = math.ceil(batches_per_epoch / config.train.accumulate_steps)
    total_steps      = steps_per_epoch * config.train.num_epochs
    trainer.module.optimizer, trainer.module.scheduler = \
        trainer.module.configure_optimizers(total_steps)

    # 7) load your two quantile‐to‐normal transformers (only rank 0 needs them for metrics)
    if rank == 0:
        with open(config.dataset.iHiTOP.valence_transform,  'rb') as f: val_tf = pickle.load(f)
        with open(config.dataset.iHiTOP.arousal_transform, 'rb') as f: aro_tf = pickle.load(f)
    else:
        val_tf = aro_tf = None


    # 8) training loop
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        train_loader.sampler.set_epoch(epoch)

        for phase in ('train', 'val'):
            loader = val_loader if phase =='val' else train_loader

            if phase=='val':
                all_out, all_gt, all_vids = [], [], []

            for batch_idx, batch in enumerate(tqdm(loader,
                                                  desc=f"Rank {rank} {phase}",
                                                  position=rank, leave=False)):
                dist.barrier()
                try:
                    # move to GPU
                    for k, v in batch.items():
                        if k in ('audio_phonemes','text_phonemes'): continue
                        if isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                            batch[k] = [x.to(rank) for x in v]
                        elif torch.is_tensor(v):
                            batch[k] = v.to(rank)

                    out = trainer.module.step(batch, batch_idx, epoch, phase=phase, ddp_model=trainer)

                    if rank == 0 and phase == 'train' and batch_idx % config.train.visualize_every == 0:
                        wandb.log({'epoch':epoch,'batch_idx':batch_idx})

                    if phase == 'val':
                        all_out.append(out['valence_arousal_out'].detach().cpu().numpy())
                        all_gt.append( out['valence_arousal_gt'].detach().cpu().numpy())
                        all_vids.extend(batch['video_ID'])

                    if rank==0 and phase == 'train' and batch_idx % 5 == 0:
                        print(out['valence_arousal_out'])
                        print(out['valence_arousal_gt'])

                except Exception:
                    if rank==0:
                        print(f"Error loading batch {batch_idx}!\n", traceback.format_exc())

                # save checkpoint
                if phase=='train' and rank==0 and (
                   batch_idx%config.train.save_every==0 or batch_idx==len(loader)-1):
                    ckpt = os.path.join(config.train.log_path, f'model_{epoch}_{batch_idx}.pt')
                    trainer.module.save_model(trainer.module.state_dict(), ckpt)

            # end batch loop

            if phase=='val':
                # 1) gather shards → full arrays
                local_out = np.concatenate(all_out,axis=0)
                local_gt  = np.concatenate(all_gt,axis=0)
                local_vid = np.array(all_vids,dtype=np.int64).flatten()

                shard_dir = os.path.join(config.train.log_path,"val_shards")
                os.makedirs(shard_dir,exist_ok=True)
                shard_path = os.path.join(shard_dir,f"epoch{epoch}_rank{rank}.npz")
                np.savez(shard_path,out=local_out,gt=local_gt,vid=local_vid)

                dist.barrier()

                if rank==0:
                    # load everything back
                    files   = sorted(f for f in os.listdir(shard_dir) if f.endswith(".npz"))
                    outs_list, gts_list, vids_list = [],[],[]
                    for fn in files:
                        d = np.load(os.path.join(shard_dir,fn))
                        outs_list.append(d["out"])
                        gts_list.append(d["gt"])
                        vids_list.append(d["vid"])
                    arr_out = np.concatenate(outs_list,axis=0)
                    arr_gt  = np.concatenate(gts_list,axis=0)
                    arr_vid = np.concatenate(vids_list,axis=0)

                    # 2) inverse‐transform both preds & GT back to original scale
                    arr_out_orig = np.column_stack([
                        val_tf.inverse_transform(arr_out[:,[0]]).flatten(),
                        aro_tf.inverse_transform(arr_out[:,[1]]).flatten()
                    ])
                    arr_gt_orig  = np.column_stack([
                        val_tf.inverse_transform(arr_gt[:,[0]]).flatten(),
                        aro_tf.inverse_transform(arr_gt[:,[1]]).flatten()
                    ])

                    # 3) compute metrics on original‐scale values
                    n_dims = arr_out_orig.shape[-1]
                    metrics_out = []

                    for i in range(n_dims):
                        y_pred = arr_out_orig[:,i]
                        y_true = arr_gt_orig[:,i]
                        mse   = float(np.mean((y_pred - y_true)**2))
                        r_all = float(metrics.pearson_r(y_pred, y_true))
                        p_all = float(metrics.pearson_p(y_pred, y_true))
                        metrics_out += [mse, r_all, p_all]

                        vids_u = np.unique(arr_vid)
                        r_w=p_w=nan_c=0.0
                        for vid in tqdm(vids_u, desc=f"Dim{i} within-video", leave=False):
                            mask = (arr_vid==vid)
                            rv = metrics.pearson_r(y_pred[mask], y_true[mask])
                            pv = metrics.pearson_p(y_pred[mask], y_true[mask])
                            if np.isnan(rv):
                                nan_c+=1
                            else:
                                r_w+=rv; p_w+=pv
                        valid = len(vids_u)-nan_c
                        if valid>0:
                            r_w/=valid; p_w/=valid
                        metrics_out += [r_w, p_w]

                        # video‐level means
                        avg_o = np.array([y_pred[arr_vid==v].mean() for v in vids_u])
                        avg_g = np.array([y_true[arr_vid==v].mean() for v in vids_u])
                        r_b = float(metrics.pearson_r(avg_o, avg_g))
                        p_b = float(metrics.pearson_p(avg_o, avg_g))
                        metrics_out += [r_b, p_b, nan_c]

                    metrics_arr = np.array(metrics_out, dtype=float)
                    print(f"[Rank 0] Epoch {epoch} metrics: {metrics_arr}")
                    np.save(os.path.join(config.train.log_path, f"metrics_{epoch}.npy"), metrics_arr)

                    # cleanup
                    for fn in files:
                        os.remove(os.path.join(shard_dir,fn))

        # end phase loop

    if rank==0:
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
