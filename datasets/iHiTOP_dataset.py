import os
import random
import sys
import torch
import pickle
import h5py
import datasets.data_utils as data_utils
import albumentations as A
from datasets.base_video_dataset import BaseVideoDataset
import numpy as np
import cv2
from multiprocessing import Pool, TimeoutError
import time


class iHiTOPDataset(BaseVideoDataset):
    def __init__(self, video_data, split_idxs, config, test=False):
        super().__init__(video_data, split_idxs, config, test)
        self.name = 'iHiTOP'
        self.num_workers = config.dataset.iHiTOP_aug_workers
        self.iHiTOP_frame_step = config.dataset.iHiTOP_frame_step

    def __len__(self):
        return len(self.split_idxs)
    
    def crop_face(self, video_dict, scale):
        w, h = video_dict["img"].shape[1:3]
        ws, hs = int(w * scale), int(h * scale)
        dw, dh = int(w - ws) // 2, (h - hs) // 2

        video_dict["img"] = video_dict["img"][:, dw:dw + ws, dh:dh + hs]
        video_dict["mask"] = video_dict["mask"][:, dw:dw + ws, dh:dh + hs]
        video_dict["landmarks_fan"] -= dw
        video_dict["landmarks_mp"] -= dw

    @staticmethod
    def process_frame(args):
        index, cropped_image, hull_mask, cropped_landmarks_fan, cropped_landmarks_mediapipe, repeat_transform, test, resize = args
        if not test:
            transformed = A.ReplayCompose.replay(repeat_transform, image=cropped_image, mask=1 - hull_mask, keypoints=cropped_landmarks_fan, mediapipe_keypoints=cropped_landmarks_mediapipe)
        else:
            transformed = resize(image=cropped_image, mask=1 - hull_mask, keypoints=cropped_landmarks_fan, mediapipe_keypoints=cropped_landmarks_mediapipe)
        return index, transformed

    def __getitem_aux__(self, index):
        v, s = self.split_idxs[index]
        video_group = self.video_data[v][str(s)]
        video_dict = {}

        # Gather all image data (subsample if necessary)
        for key, item in video_group.items():
            value = item[()]
            if isinstance(value, np.ndarray):
                sampled_value = value.copy()  # Copy the data to avoid issues with multiprocessing
                sampled_idxs = np.arange(len(sampled_value), step=self.iHiTOP_frame_step).astype(int)
                sampled_value = sampled_value[sampled_idxs]

                video_dict[key] = sampled_value

        # Resize data based on random or fixed scale
        if isinstance(self.scale, list):
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
            scale = scale / self.prescale
        else:
            scale = self.scale / self.prescale
        
        self.crop_face(video_dict, scale)

        # Apply data augmentations
        new_imgs = []
        new_fan_lmks = []
        new_mp_lmks = []
        new_masks = []

        if not self.test:
            repeat_transform = self.transform(
                image=video_dict["img"][0],
                mask=1 - video_dict["mask"][0],
                keypoints=video_dict["landmarks_fan"][0],
                mediapipe_keypoints=video_dict["landmarks_mp"][0]
            )["replay"]

        # Properly pass all arguments for multiprocessing
        args_list = [(i, video_dict["img"][i], video_dict["mask"][i], video_dict["landmarks_fan"][i],
                      video_dict["landmarks_mp"][i], repeat_transform, self.test, self.resize) 
                      for i in range(len(video_dict["img"]))]

        # Use apply_async with a timeout and maintain order
        results = []
        with Pool(self.num_workers) as pool:
            async_results = [pool.apply_async(self.process_frame, args=(arg,)) for arg in args_list]
            
            for res in async_results:
                try:
                    # Set a timeout of 30 seconds for each task
                    result = res.get(timeout=30)
                    results.append(result)
                except TimeoutError:
                    print(f"{index} timed out and was skipped.")
                    continue

        # Sort results by the original index to maintain the order
        results = sorted(results, key=lambda x: x[0])

        # Unpack the sorted results
        for _, transformed in results:
            new_imgs.append((transformed['image'] / 255.0).astype(np.float32).transpose(2, 0, 1)[None])
            new_fan_lmks.append(np.array(transformed['keypoints']).astype(np.float32)[None])
            new_mp_lmks.append(np.array(transformed['mediapipe_keypoints']).astype(np.float32)[None])
            new_masks.append((1 - transformed['mask'])[..., None].transpose(2, 0, 1)[None])

        video_dict["img"] = np.concatenate(new_imgs)
        video_dict["landmarks_fan"] = np.concatenate(new_fan_lmks)
        video_dict["landmarks_mp"] = np.concatenate(new_mp_lmks)
        video_dict["mask"] = np.concatenate(new_masks)

        # Misc processing
        video_dict["landmarks_fan"][:, :, :2] = video_dict["landmarks_fan"][:, :, :2] / self.image_size * 2 - 1
        video_dict["landmarks_mp"][:, :, :2] = video_dict["landmarks_mp"][:, :, :2] / self.image_size * 2 - 1
        video_dict["mask"] = video_dict["mask"].astype(np.float32)

        # Turning numpy arrays into tensors
        for key, item in video_dict.items():
            if isinstance(item, np.ndarray) and item.dtype != "S1":
                video_dict[key] = torch.tensor(item)

        return video_dict

def cull_indices(hdf5_files, seg_indices, config):
    valid_mask = np.ones(len(seg_indices), dtype=bool)  # Start with all indices valid

    # Loop over the seg_indices and check against the threshold
    for i, (file_idx, group_idx) in enumerate(seg_indices):
        try:
            file_idx = int(file_idx)
            group_idx = str(int(group_idx))  # Convert group index to string
            
            hdf5_file = hdf5_files[file_idx]
            group = hdf5_file[group_idx]
            
            img_length = len(group["img"])
            removed_frames_length = len(group["removed_frames"])
            total_length = img_length + removed_frames_length
            
            # Calculate removed frames percentage
            if total_length > 0:  # Avoid division by zero
                removed_frames_percentage = removed_frames_length / total_length
            else:
                removed_frames_percentage = 0

            # If the percentage exceeds the threshold, mark the index as invalid
            if removed_frames_percentage >= config.dataset.removed_frames_threshold:
                valid_mask[i] = False  # Mark invalid if threshold is exceeded

            # Also, remove if img_length is too long
            if img_length > config.dataset.max_seg_len:
                valid_mask[i] = False 

            # Or too short
            if img_length < config.dataset.min_seg_len:
                valid_mask[i] = False 
        except:
            print(f"Segment {group_idx} failed from {file_idx}!")
            valid_mask[i] = False
    
    return valid_mask  # Return the boolean mask

def get_datasets_iHiTOP(config=None):
    # Gather all hdf5 files and number of segments in each
    hdf5_files = []
    seg_counts = []
    bad_files = []
    for file in os.listdir(config.dataset.iHiTOP_hdf5_path):
        hdf5_file = None
        try:
            hdf5_file = h5py.File(os.path.join(
                config.dataset.iHiTOP_hdf5_path,
                file
            ), 'r')
            seg_count = len(hdf5_file.keys())

            hdf5_files.append(hdf5_file)
            seg_counts.append(seg_count)
        except:
            print(f"Bad File {file}")
            bad_files.append(file)
            if hdf5_file is not None:
                hdf5_file.close()
    
    if not os.path.exists(config.dataset.bad_files):
        np.save(config.dataset.bad_files, np.array(bad_files))

    seg_indices = []
    # Set up indexing to access all of these files
    for i, hdf5_file in enumerate(hdf5_files):
        for key in hdf5_file.keys():
            seg_indices.append(np.array([i, int(key)])[None])
    seg_indices = np.concatenate(seg_indices)

    # Now cull indices depending on their number of removed frames
    if not os.path.exists(config.dataset.data_idxs):
        print("Data indices undefined! Defining now....")
        np.save(config.dataset.data_idxs, cull_indices(hdf5_files, seg_indices, config))
        print("Data indices defined!")

    data_idxs = np.load(config.dataset.data_idxs)
    seg_indices = seg_indices[data_idxs]
    print(f"Segment count {len(seg_indices)}")

    # Assert train, val, test split
    assert config.dataset.iHiTOP_train_percentage + config.dataset.iHiTOP_val_percentage + config.dataset.iHiTOP_test_percentage == 1.0
    total = len(seg_indices)

    train_size = int(config.dataset.iHiTOP_train_percentage * total)
    val_size = int(config.dataset.iHiTOP_val_percentage * total)
    test_size = total - train_size - val_size

    # this is the split used in the paper, randomly selected
    random_idxs = seg_indices
    np.random.shuffle(random_idxs)

    if os.path.exists(config.dataset.final_idxs):
        random_idxs = np.load(config.dataset.final_idxs)
    else:
        random_idxs = seg_indices
        np.random.shuffle(random_idxs)
        print("Shuffled indices undefined! Defining now....")
        np.save(config.dataset.final_idxs, random_idxs)
        print("Shuffled indices defined!")

    train_idxs = random_idxs[:train_size]
    val_idxs = random_idxs[train_size:train_size + val_size]
    test_idxs = random_idxs[train_size + val_size:]

    return iHiTOPDataset(hdf5_files, train_idxs, config), iHiTOPDataset(hdf5_files, val_idxs, config, test=True), iHiTOPDataset(hdf5_files, test_idxs, config, test=True) #, train_list, val_list, test_list
