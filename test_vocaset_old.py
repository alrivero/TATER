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
from datasets.EXT_dataset import get_datasets_VOCASET
from datasets.mixed_dataset_sampler import MixedDatasetBatchSampler
import pickle
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds

def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

def load_dataloaders(config):
    # ----------------------- initialize datasets ----------------------- #
    train_dataset_VOCASET, val_dataset_VOCASET, test_dataset_VOCASET = get_datasets_VOCASET(config)
    dataset_percentages = {
        'VOCASET': 1.0
    }
    
    train_dataset = train_dataset_VOCASET
    sampler = MixedDatasetBatchSampler([
                                        len(train_dataset_VOCASET)
                                        ], 
                                       list(dataset_percentages.values()), 
                                       config.train.batch_size, len(train_dataset_VOCASET))
    def collate_fn(batch):
        combined_batch = {}
        for key in batch[0].keys():
            combined_batch[key] = [b[key] for b in batch]

        return combined_batch
    
    val_dataset = val_dataset_VOCASET
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=config.train.batch_size, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_dataset_VOCASET

if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    warnings.filterwarnings("ignore", message="GaussNoise could work incorrectly in ReplayMode for other input data")
    _, val_dataloader,  _ = load_dataloaders(config)

    def strip_exact_prefix(state_dict, prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # Initialize models
    tater = TATEREncoder(config, n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    loaded_state_dict = torch.load(config.resume, map_location=config.device)
    filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k.startswith('tater')}
    filtered_state_dict = strip_exact_prefix(filtered_state_dict, "tater.")
    tater.load_state_dict(filtered_state_dict, strict=False)
    tater.device = config.device
    force_model_to_device(tater, tater.device)
    tater.eval()

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
