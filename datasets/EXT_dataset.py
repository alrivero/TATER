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

class EXTDataset(BaseVideoDataset):
    def __init__(self, video_data, split_idxs, config, test=False):
        super().__init__(video_data, split_idxs, config, test)
        self.name = 'EXT'

    def __len__(self):
        return len(self.split_idxs)

    def __getitem_aux__(self, index):
        # Convert data_dir string to a Path object
        data_dir = Path(self.video_data[self.split_idxs[index]])  # Ensure this is a Path object and not an int

        # Get sorted .jpg files and .obj files in the directory
        img_files = sorted([f for f in data_dir.iterdir() 
                            if f.is_file() and f.suffix == '.png'])

        # Load and resize images
        images = [np.array(Image.open(file)) for file in img_files]
        images_array = np.array(images)

        # Load audio data from .wav file associated with data_dir
        wav_path = data_dir.with_suffix('.wav')
        sample_rate, audio_data = wavfile.read(str(wav_path))  # Convert to str to ensure compatibility
        
        # Construct the data dictionary, only converting actual data (not paths) to tensors
        data = {
            "img": torch.tensor(images_array).float().permute(0, 3, 1, 2) / 255,
            "audio": torch.tensor(audio_data),
            "audio_sample_rate": torch.tensor([sample_rate]).float(),
            "fps": torch.tensor([25]).float()
        }

        return data

def get_datasets_EXT(config=None):
    main_dir = config.dataset.EXT.data_path
    directories = [f"{main_dir}/{d}" for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

    # Assert train, val, test split
    assert config.dataset.EXT.train_percentage + config.dataset.EXT.val_percentage + config.dataset.EXT.test_percentage == 1.0
    total = len(directories)

    train_size = int(config.dataset.EXT.train_percentage * total)
    val_size = int(config.dataset.EXT.val_percentage * total)
    test_size = total - train_size - val_size

    idxs = np.arange(total)

    train_idxs = idxs[:train_size]
    val_idxs = idxs[train_size:train_size + val_size]
    test_idxs = idxs[train_size + val_size:]

    return EXTDataset(directories, train_idxs, config, test=True), EXTDataset(directories, val_idxs, config, test=True), EXTDataset(directories, test_idxs, config, test=True)
