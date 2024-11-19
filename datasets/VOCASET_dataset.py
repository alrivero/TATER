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

class VOCASETDataset(BaseVideoDataset):
    def __init__(self, video_data, split_idxs, config, test=False):
        super().__init__(video_data, split_idxs, config, test)
        self.name = 'VOCASET'

    def __len__(self):
        return len(self.split_idxs)

    def __getitem_aux__(self, index):
        # Convert data_dir string to a Path object
        data_dir = Path(self.video_data[self.split_idxs[index]])  # Ensure this is a Path object and not an int

        # Get sorted .jpg files and .obj files in the directory
        img_files = sorted([f for f in data_dir.iterdir() 
                            if f.is_file() and f.suffix == '.jpg' and f.stem.endswith('26_C')])
        obj_files = sorted([f for f in data_dir.iterdir() 
                            if f.is_file() and f.suffix == '.obj'])

        # Load and resize images
        images = [np.array(Image.open(file).resize((224, 224))) for file in img_files]
        images_array = np.array(images)

        # Load vertices from each .obj file using trimesh
        vertices_list = [torch.tensor(trimesh.load(str(file)).vertices).float() for file in obj_files]  # Convert each file to str if needed
        for i, verts in enumerate(vertices_list):
            vertices_list[i] = verts - verts.min(dim=0)[0]
            vertices_list[i] = verts / verts.max(dim=0)[0]

        # Load audio data from .wav file associated with data_dir
        wav_path = data_dir.with_suffix('.wav')
        sample_rate, audio_data = wavfile.read(str(wav_path))  # Convert to str to ensure compatibility
        
        # Construct the data dictionary, only converting actual data (not paths) to tensors
        data = {
            "img": torch.tensor(images_array).float().permute(0, 3, 1, 2),
            "verts": vertices_list,
            "audio": torch.tensor(audio_data),
            "audio_sample_rate": torch.tensor([sample_rate]).float(),
            "fps": torch.tensor([30]).float()
        }

        return data

def get_datasets_VOCASET(config=None):
    main_dir = config.dataset.VOCASET.data_path
    directories = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

    data_dirs = []
    for dir in directories:
        # Construct the full path of the current directory
        dir_path = os.path.join(main_dir, dir)
        
        # Find all subdirectories and store their full paths
        sub_directories = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) for f in os.listdir(os.path.join(dir_path, d)))]
        data_dirs += sub_directories


    # Assert train, val, test split
    assert config.dataset.VOCASET.train_percentage + config.dataset.VOCASET.val_percentage + config.dataset.VOCASET.test_percentage == 1.0
    total = len(data_dirs)

    train_size = int(config.dataset.VOCASET.train_percentage * total)
    val_size = int(config.dataset.VOCASET.val_percentage * total)
    test_size = total - train_size - val_size

    # this is the split used in the paper, randomly selected
    if os.path.exists(config.dataset.VOCASET.final_idxs):
        random_idxs = np.load(config.dataset.VOCASET.final_idxs)
    else:
        random_idxs = np.arange(total)
        np.random.shuffle(random_idxs)
        print("Shuffled indices undefined! Defining now....")
        np.save(config.dataset.VOCASET.final_idxs, random_idxs)
        print("Shuffled indices defined!")

    train_idxs = random_idxs[:train_size]
    val_idxs = random_idxs[train_size:train_size + val_size]
    test_idxs = random_idxs[train_size + val_size:]

    return VOCASETDataset(data_dirs, train_idxs, config, test=True), VOCASETDataset(data_dirs, val_idxs, config, test=True), VOCASETDataset(data_dirs, test_idxs, config, test=True)
