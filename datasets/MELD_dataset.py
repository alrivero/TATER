import os
import torch
import h5py
import albumentations as A
from datasets.base_video_dataset import BaseVideoDataset
from scipy.io import wavfile
from PIL import Image
import numpy as np
import trimesh
from pathlib import Path
import h5py

class MELDDataset(BaseVideoDataset):
    def __init__(self, video_data, split_idxs, config, test=False):
        super().__init__(video_data, split_idxs, config, test)
        self.name = 'MELD'
        self.emotion_map = {
            "neutral": 0,
            "surprise": 1,
            "fear": 2,
            "joy": 3,
            "sadness": 4,
            "disgust": 5,
            "anger": 6,
        }

    def __len__(self):
        return len(self.split_idxs)

    def __getitem_aux__(self, index):
        key = self.split_idxs[index]
        video_group = self.video_data[key]
        data = {}

        data["img"] = torch.tensor(video_group["frames"])
        data["audio"] = torch.tensor(video_group["audio"])
        data["fps"] = video_group.attrs["fps"]
        data["sample_rate"] = video_group.attrs["sample_rate"]
        data["emotion"] = self.emotion_map[video_group.attrs["Emotion"]]

        return data

def get_datasets_MELD(config=None):
    video_data = h5py.File(config.dataset.MELD.data_file, 'r')
    group_keys = np.array(list(video_data.keys()))

    # Assert train, val, test split
    assert config.dataset.MELD.train_percentage + config.dataset.MELD.val_percentage + config.dataset.MELD.test_percentage == 1.0
    total = len(group_keys)

    train_size = int(config.dataset.MELD.train_percentage * total)
    val_size = int(config.dataset.MELD.val_percentage * total)
    test_size = total - train_size - val_size

    # this is the split used in the paper, randomly selected
    if os.path.exists(config.dataset.MELD.final_idxs):
        random_idxs = np.load(config.dataset.MELD.final_idxs)
    else:
        random_idxs = np.arange(total)
        np.random.shuffle(random_idxs)
        print("Shuffled indices undefined! Defining now....")
        np.save(config.dataset.MELD.final_idxs, random_idxs)
        print("Shuffled indices defined!")

    train_idxs = group_keys[random_idxs[:train_size]]
    val_idxs = group_keys[random_idxs[train_size:train_size + val_size]]
    test_idxs = group_keys[random_idxs[train_size + val_size:]]

    return MELDDataset(video_data, train_idxs, config, test=True), MELDDataset(video_data, val_idxs, config, test=True), MELDDataset(video_data, test_idxs, config, test=True)
