import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import os
import debug
import traceback
import time
import warnings
from src.tater_encoder import TATEREncoder
from src.FLAME.FLAME import FLAME
from datasets.VOCASET_dataset import get_datasets_VOCASET
from datasets.mixed_dataset_sampler import MixedDatasetBatchSampler
import pickle
from pytorch3d.loss import chamfer_distance

def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

def load_dataloaders(config):
    # ----------------------- initialize datasets ----------------------- #
    train_dataset_iHiTOP, val_dataset_iHiTOP, test_dataset_iHiTOP = get_datasets_VOCASET(config)
    dataset_percentages = {
        'iHiTOP': 1.0
    }
    
    train_dataset = train_dataset_iHiTOP
    sampler = MixedDatasetBatchSampler([
                                        len(train_dataset_iHiTOP)
                                        ], 
                                       list(dataset_percentages.values()), 
                                       config.train.batch_size, len(train_dataset_iHiTOP))
    def collate_fn(batch):
        combined_batch = {}
        for key in batch[0].keys():
            combined_batch[key] = [b[key] for b in batch]

        return combined_batch
    
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_iHiTOP])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=config.train.batch_size, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_dataset_iHiTOP

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

if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    warnings.filterwarnings("ignore", message="GaussNoise could work incorrectly in ReplayMode for other input data")
    val_loader, _,  _ = load_dataloaders(config)

    tater = TATEREncoder(config, n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    tater = tater.to(config.device)

    flame = FLAME(n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    flame = flame.to(config.device)
    flame_masks = pickle.load(open("/home/arivero/FLAME_masks.pkl", "rb"), encoding="latin1")

    def strip_exact_prefix(state_dict, prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    
    loaded_state_dict = torch.load(config.resume, map_location=config.device)
    filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k.startswith('tater')}
    filtered_state_dict = strip_exact_prefix(filtered_state_dict, "tater.")
    tater.load_state_dict(filtered_state_dict)
    tater.eval()

    all_chamfer = []
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):        
        for phase in ['val']:
            loader = val_loader
            
            for segment_idx, batch in tqdm(enumerate(loader), total=len(loader)):
                if len(batch["img"][0]) == 0:
                    print("Empty. Continue...")
                    continue

                framewise_keys = ["img", "verts"]
                batchwise_keys = ["audio_sample_rate", "fps"]
                split_batches = batch_split_into_windows(batch, config, framewise_keys, batchwise_keys)

                for split_batch in split_batches:
                    for key in split_batch:
                            if torch.is_tensor(split_batch[key][0]):
                                for i in range(len(split_batch[key])):
                                    split_batch[key][i] = split_batch[key][i].to(config.device)

                    with torch.no_grad():
                        series_len = [b.shape[0] for b in split_batch["img"]]
                        all_params = tater(split_batch["img"][0], series_len)

                        meshes = flame.forward(all_params)["vertices"]
                        meshes = meshes - meshes.min(dim=1)[0][:, None, :]
                        meshes = meshes / meshes.max(dim=1)[0][:, None, :]

                        for mesh, gt in zip(meshes, split_batch["verts"][0]):
                            chamfer_dist = chamfer_distance(mesh[None], gt[None].to(config.device).float())
                            print(f"Chamfer Distance: {chamfer_dist[0]}")
                            all_chamfer.append(chamfer_dist[0])
