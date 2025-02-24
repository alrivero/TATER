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
import torchaudio
import torchaudio.transforms as T
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from itertools import cycle

class VOCASETDataset(BaseVideoDataset):
    def __init__(self, video_data, split_idxs, config, test=False):
        super().__init__(video_data, split_idxs, config, test)
        self.name = 'VOCASET'
        self.original_fps = 60  # VOCASET dataset is at 60 FPS
        self.target_fps = config.get('target_fps', 25)  # Allow user-defined target FPS

    def __len__(self):
        return len(self.split_idxs)

    def find_exhaustive_neighbor(self, index, available_indices):
        """Finds the closest valid left/right neighbor, checking exhaustively if necessary."""
        if index in available_indices:
            return index  # Direct hit

        # If no valid indices exist, return None (handle it safely later)
        if not available_indices:
            return None

        # Search outward for the closest available index
        offset = 1
        while True:
            left = index - offset
            right = index + offset

            if left in available_indices:
                return left
            if right in available_indices:
                return right

            # If we exceed the available range, return the closest bound
            if left < min(available_indices) and right > max(available_indices):
                return min(available_indices) if abs(left - index) < abs(right - index) else max(available_indices)

            # Prevent infinite loops by capping the search range
            if offset > (max(available_indices) - min(available_indices)):
                return min(available_indices)  # Return smallest valid index as fallback

            offset += 1  # Expand search outward

    def __getitem_aux__(self, index):
        # **Use index to get data directory**
        data_dir = Path(self.video_data[self.split_idxs[index]])
        problematic = False  # Track issues

        # **Get all valid .jpg and .obj files**
        img_files = sorted([f for f in data_dir.iterdir() if f.suffix == '.jpg' and f.stem.endswith('26_C')])
        obj_files = sorted([f for f in data_dir.iterdir() if f.suffix == '.obj'])

        # **Extract numerical indices from file names**
        img_indices = {int(f.stem.split('.')[1]) for f in img_files}
        obj_indices = {int(f.stem.split('.')[1]) for f in obj_files}
        
        valid_indices = sorted(img_indices & obj_indices)  # Keep only indices present in both sets

        # **Raise an exception if all frames are missing**
        if not valid_indices:
            raise ValueError(f"Missing frames in {data_dir}")

        # **Compute Sampling Indices Safely**
        # Compute Sampling Indices Safely
        max_index = max(valid_indices)
        sampling_ratio = self.original_fps / self.target_fps
        sampled_indices = np.round(np.arange(0, max_index + 1, sampling_ratio)).astype(int)

        # Ensure `sampled_indices` do not exceed valid range
        sampled_indices = [i for i in sampled_indices if min(valid_indices) <= i <= max(valid_indices)]
        sampled_indices = [self.find_exhaustive_neighbor(i, valid_indices) for i in sampled_indices]

        # **Create lookup dictionaries for images and objects**
        img_lookup = {int(f.stem.split('.')[1]): f for f in img_files}
        obj_lookup = {int(f.stem.split('.')[1]): f for f in obj_files}

        # **Load and resize sampled images (Use closest L/R neighbor if missing)**
        images = []

        for i in sampled_indices:
            valid_i = self.find_exhaustive_neighbor(i, img_lookup.keys())
            if valid_i is None:
                continue  # Skip missing frame safely
            img_file = img_lookup.get(valid_i)

            if img_file:
                try:
                    img = Image.open(img_file)
                    img_array = np.array(img) / 255.0  # Normalize
                except Exception as e:
                    print(f"Corrupt image at {img_file}, using another neighbor. Error: {e}")
                    problematic = True
                    valid_i = self.find_exhaustive_neighbor(valid_i, img_lookup.keys())  # Find another valid neighbor
                    img_file = img_lookup.get(valid_i)
                    img = Image.open(img_file)  # Load neighbor frame
                    img_array = np.array(img) / 255.0  # Normalize
            else:
                raise RuntimeError(f"No valid image found for index {i} or any neighbor.")

            images.append(img_array)

        # Convert to NumPy array
        images_array = np.array(images)

        # **Load GT Meshes One-by-One (Use closest L/R neighbor if missing)**
        valid_meshes = []

        for i in sampled_indices:
            valid_i = self.find_exhaustive_neighbor(i, obj_lookup.keys())
            if valid_i is None:
                continue  # Skip missing frame safely
            obj_file = obj_lookup.get(valid_i)

            if obj_file:
                try:
                    mesh = load_objs_as_meshes([obj_file], device="cpu")[0]  # Load mesh individually
                    verts = mesh.verts_packed()

                    if verts.shape[0] == 0:  # Mesh has zero vertices
                        print(f"Zero-sized mesh at {obj_file}, searching further.")
                        problematic = True
                        valid_i = self.find_exhaustive_neighbor(valid_i, obj_lookup.keys())  # Find another valid neighbor
                        obj_file = obj_lookup.get(valid_i)
                        mesh = load_objs_as_meshes([obj_file], device="cpu")[0]  # Load neighbor mesh
                    
                    # **Normalize the mesh to fit within a unit sphere**
                    centroid = verts.mean(dim=0, keepdim=True)
                    verts -= centroid  # Shift to origin
                    max_dist = torch.linalg.norm(verts, dim=1).max()
                    verts /= max_dist  # Normalize
                    mesh._verts_list[0] = verts  # Update mesh

                except Exception as e:
                    print(f"Corrupt mesh at {obj_file}, searching further. Error: {e}")
                    problematic = True
                    valid_i = self.find_exhaustive_neighbor(valid_i, obj_lookup.keys())  # Find another valid neighbor
                    obj_file = obj_lookup.get(valid_i)
                    mesh = load_objs_as_meshes([obj_file], device="cpu")[0]  # Load neighbor mesh
            else:
                raise RuntimeError(f"No valid mesh found for index {i} or any neighbor.")

            valid_meshes.append(mesh)

        # **Load and Resample Audio (NO TRY-EXCEPT)**
        wav_path = data_dir.with_suffix('.wav')
        sample_rate, audio_data = wavfile.read(str(wav_path))  # Load audio
        audio_data = torch.tensor(audio_data).float()
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(dim=-1)  # Convert to mono
        
        resample_ratio = self.target_fps / self.original_fps
        new_audio_sample_rate = int(sample_rate * resample_ratio)
        resampler = T.Resample(orig_freq=sample_rate, new_freq=new_audio_sample_rate)
        audio_data = resampler(audio_data.unsqueeze(0)).squeeze(0)  # Resample

        # **Construct the data dictionary**
        data = {
            "img": torch.tensor(images_array).float().permute(0, 3, 1, 2),
            "meshes": valid_meshes if valid_meshes else None,
            "audio": audio_data,
            "audio_sample_rate": torch.tensor([new_audio_sample_rate]).float(),
            "fps": torch.tensor([self.target_fps]).float(),
            "data_dir": data_dir,
            "problematic": problematic  # Return the flag
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


    if not os.path.exists("randomized_indices_VOCASET.npy"):
        total = len(data_dirs)
        all_idxs = np.arange(total)
        np.random.shuffle(all_idxs)
        np.save("randomized_indices_VOCASET.npy", all_idxs)
    else:
        all_idxs = np.load("randomized_indices_VOCASET.npy")

    train_idxs = []
    val_idxs = all_idxs[:int(0.25 * len(data_dirs))]
    test_idxs = []

    return VOCASETDataset(data_dirs, train_idxs, config, test=True), VOCASETDataset(data_dirs, val_idxs, config, test=True), VOCASETDataset(data_dirs, test_idxs, config, test=True)
