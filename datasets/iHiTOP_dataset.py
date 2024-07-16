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
from multiprocessing import Pool
import time


class iHiTOPDataset(BaseVideoDataset):
    def __init__(self, video_data, split_idxs, config, test=False):
        super().__init__(video_data, split_idxs, config, test)
        self.name = 'iHiTOP'
        self.dataset_size = config.dataset.iHiTOP_dataset_size
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
        video_dict["landmarks_fan"] -= (2 * ws)
        video_dict["landmarks_mp"] -= (2 * ws)

    @staticmethod
    def process_frame(args):
        cropped_image, hull_mask, cropped_landmarks_fan, cropped_landmarks_mediapipe, repeat_transform, test, resize = args
        if not test:
            transformed = A.ReplayCompose.replay(repeat_transform, image=cropped_image, mask=1 - hull_mask, keypoints=cropped_landmarks_fan, mediapipe_keypoints=cropped_landmarks_mediapipe)
        else:
            transformed = resize(image=cropped_image, mask=1 - hull_mask, keypoints=cropped_landmarks_fan, mediapipe_keypoints=cropped_landmarks_mediapipe)
        return transformed

    def __getitem_aux__(self, index):
        video_group = self.video_data[str(self.split_idxs[index])]
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
            # select randomly a zoom scale during training for cropping
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
            repeat_transform = self.transform(image=video_dict["img"][0], mask=1 - video_dict["mask"][0], keypoints=video_dict["landmarks_fan"][0], mediapipe_keypoints=video_dict["landmarks_mp"][0])["replay"]

        args_list = []
        for i in range(len(video_dict["img"])):
            args_list.append((video_dict["img"][i], video_dict["mask"][i], video_dict["landmarks_fan"][i], video_dict["landmarks_mp"][i], repeat_transform, self.test, self.resize))

        with Pool(self.num_workers) as pool:
            results = pool.map(self.process_frame, args_list)

        for transformed in results:
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
            if isinstance(item, np.ndarray):
                video_dict[key] = torch.tensor(item)

        return video_dict


def get_datasets_iHiTOP(config=None):
    # Assuming you're currently in the directory where the files are located
    video_data = h5py.File(config.dataset.iHiTOP_hdf5_path, 'r')

    assert config.dataset.iHiTOP_train_percentage + config.dataset.iHiTOP_val_percentage + config.dataset.iHiTOP_test_percentage == 1.0
    total = config.dataset.iHiTOP_dataset_size

    train_size = int(config.dataset.iHiTOP_train_percentage * total)
    val_size = int(config.dataset.iHiTOP_val_percentage * total)
    test_size = total - train_size - val_size

    # this is the split used in the paper, randomly selected
    random_idxs = np.arange(total)
    np.random.shuffle(random_idxs)

    train_idxs = random_idxs[:train_size]
    val_idxs = random_idxs[train_size:train_size + val_size]
    test_idxs = random_idxs[train_size + val_size:]

    return iHiTOPDataset(video_data, train_idxs, config), iHiTOPDataset(video_data, val_idxs, config, test=True), iHiTOPDataset(video_data, test_idxs, config, test=True) #, train_list, val_list, test_list

