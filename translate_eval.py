import sys
from omegaconf import OmegaConf
import argparse
import torch
from tqdm import tqdm
import os
import debug
import traceback
import time
import warnings
from datasets.data_utils import load_dataloaders
from src.tater_encoder import TATEREncoder
from src.translate_trainer import TranslateTrainer
from src.FLAME.FLAME import FLAME
from datasets.VOCASET_dataset import get_datasets_VOCASET
from datasets.mixed_dataset_sampler import MixedDatasetBatchSampler
import pickle

def parse_args():
    # Add argparse for additional CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",  # Positional argument for the config file
        type=str,
        help="Path to the configuration YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TATER",
        help="Model currently being evaluated"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Model checkpoint to use for evaluation"
    )
    cli_args = parser.parse_args()

    # Load configuration from the provided config file
    conf = OmegaConf.load(cli_args.config_path)

    # Merge argparse arguments with OmegaConf
    conf = OmegaConf.merge(conf, vars(cli_args))
    return conf


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
    train_loader, _, effective_seg_count = load_dataloaders(config)

    if config.model == "TATER":
        model = TATEREncoder(config, n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
        model = model.to(config.device)

        def strip_exact_prefix(state_dict, prefix):
                return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        loaded_state_dict = torch.load(config.checkpoint, map_location=config.device)
        filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k.startswith('tater')}
        filtered_state_dict = strip_exact_prefix(filtered_state_dict, "tater.")
        model.load_state_dict(filtered_state_dict)
        model.eval()

    trainer = TranslateTrainer(config)
    trainer = trainer.to(config.device)
    trainer.configure_optimizers(effective_seg_count)

    all_chamfer = []
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):   

        batch_idx = 0     
        for phase in ['val']:
            loader = train_loader
            
            for segment_idx, batch in tqdm(enumerate(loader), total=len(loader)):
                try:
                    if len(batch["img"][0]) == 0:
                        print("Empty. Continue...")
                        continue

                    framewise_keys = ["flag_landmarks_fan", "img", "img_mica", "landmarks_fan", "landmarks_mp", "mask"]
                    batchwise_keys = ["text", "fps", "audio_sample_rate", "silent", "text_phonemes"]
                    extra_keys = [
                        "audio_phonemes",
                        "text_phonemes",
                        "og_idxs",
                        "removed_frames",
                        "roberta",
                        "roberta_pca_5",
                        "valence_arousal",
                        "hubert_frame_level_embeddings",
                        "wav2vec2_frame_level_embeddings",
                        "phoneme_timestamps",
                        "text_phonemes"
                    ]
                    split_batches = batch_split_into_windows(batch, config, framewise_keys, batchwise_keys)

                    all_outputs = []
                    for split_batch in split_batches:
                        for key in split_batch:
                            if key not in extra_keys and torch.is_tensor(split_batch[key][0]):
                                for i in range(len(split_batch[key])):
                                    split_batch[key][i] = split_batch[key][i].to(config.device)

                        with torch.no_grad():
                            series_len = [b.shape[0] for b in split_batch["img"]]
                            all_params = model(split_batch["img"][0], series_len)
                        all_outputs.append(trainer.step(split_batch, all_params, batch_idx, epoch, phase))

                        for key in split_batch:
                            if key not in extra_keys and torch.is_tensor(split_batch[key][0]):
                                for i in range(len(split_batch[key])):
                                    split_batch[key][i] = split_batch[key][i].to("cpu")
                    
                        batch_idx += 1

                except Exception as e:
                    print(f"Error loading batch_idx {batch_idx}!")
                    print(e)
                    traceback.print_exc()

                if batch_idx % config.train.save_every == 0 or batch_idx == len(loader) - 1:
                    trainer.save_model(trainer.state_dict(), os.path.join(config.train.log_path, f'model_{epoch}_{batch_idx}.pt'))

