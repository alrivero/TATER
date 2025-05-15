import os
import random
import sys
import torch
import pickle
import h5py
import datasets.data_utils as data_utils
import albumentations as A
from datasets.base_video_dataset import BaseVideoDataset
from transformers import Wav2Vec2Processor
import numpy as np
import cv2
from multiprocessing import Pool, TimeoutError
import time
import math
import debug
from torch.distributed import barrier, get_rank
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import skewnorm


class iHiTOPDatasetParallel(BaseVideoDataset):
    def __init__(self, hdf5_paths, split_idxs, config, test=False):
        super().__init__(None, split_idxs, config, test)
        self.name = 'iHiTOP'
        self.hdf5_paths = hdf5_paths  # Store paths in the dataset
        self.split_idxs = split_idxs
        self.iHiTOP_frame_step = config.dataset.iHiTOP.frame_step
        self.phoneme_map = self.create_phoneme_map()
        self.worker_hdf5_handles = None  # Initialize empty handles

        self.framewise_keys = ["flag_landmarks_fan", "img", "img_mica", "landmarks_fan", "landmarks_mp", "mask", "hubert_frame_level_embeddings_v4"]

    def _get_hdf5_file(self, file_idx):
        """Fetch the HDF5 file handle for the given index."""
        if self.worker_hdf5_handles is None or file_idx not in self.worker_hdf5_handles:
            raise RuntimeError(f"HDF5 file for index {file_idx} not loaded in this worker. (Total: {len(self.worker_hdf5_handles)})")
        return self.worker_hdf5_handles[file_idx]
    
    def initialize_worker_hdf5_handles(self):
        """Initialize HDF5 handles for this worker."""
        self.worker_hdf5_handles = {}
        for idx, path in enumerate(self.hdf5_paths):
            # print(path)
            self.worker_hdf5_handles[idx] = h5py.File(path, 'r')  # Open files in read mode
        print(f"Worker {torch.utils.data.get_worker_info().id} initialized {len(self.worker_hdf5_handles)} HDF5 handles.")

    def initialize_wav2vec_worker_hdf5_handles(self):
        """Initialize HDF5 handles for this worker, handling missing files."""
        self.wav2vec_worker_hdf5_handles = {}
        
        for idx, path in enumerate(self.hdf5_paths):
            # Extract the parent directory and filename
            parent_dir, file_name = os.path.split(path)
            grandparent_dir, last_parent_dir = os.path.split(parent_dir)

            # Modify last parent directory and file name
            new_last_parent_dir = f"{last_parent_dir}_wav2vec_asr_2"
            new_file_name = f"wav2vec_{file_name}"

            # Construct the new full path
            new_parent_path = os.path.join(grandparent_dir, new_last_parent_dir)
            new_path = os.path.join(new_parent_path, new_file_name)

            # Ensure the new directory exists
            os.makedirs(new_parent_path, exist_ok=True)

            # Check if file exists before opening
            if os.path.exists(new_path):
                try:
                    self.wav2vec_worker_hdf5_handles[idx] = h5py.File(new_path, 'r')
                except OSError as e:
                    print(f"Warning: Unable to open HDF5 file {new_path}. Error: {e}")
            else:
                print(f"Warning: Wav2Vec HDF5 file {new_path} does not exist. Skipping.")

        print(f"Worker {torch.utils.data.get_worker_info().id} initialized {len(self.wav2vec_worker_hdf5_handles)} Wav2VecHDF5 handles.")

    def _get_wav2vec_hdf5_file(self, file_idx):
        """Fetch the HDF5 file handle for the given index, handling missing files."""
        if not self.wav2vec_worker_hdf5_handles:
            raise RuntimeError("No Wav2Vec HDF5 files have been loaded in this worker.")

        if file_idx not in self.wav2vec_worker_hdf5_handles:
            raise RuntimeError(
                f"Wav2Vec HDF5 file for index {file_idx} is not available in this worker. "
                f"Loaded files: {list(self.wav2vec_worker_hdf5_handles.keys())}"
            )
        
        return self.wav2vec_worker_hdf5_handles[file_idx]

    def __del__(self):
        """Ensure HDF5 file handles are properly closed."""
        if self.worker_hdf5_handles is not None:
            for handle in self.worker_hdf5_handles.values():
                handle.close()

    def hdf5_worker_init(worker_id):
        """Initialize worker-specific HDF5 file handles."""
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # Access the dataset
        dataset.initialize_worker_hdf5_handles()
        dataset.initialize_wav2vec_worker_hdf5_handles()

    def create_phoneme_map(self):
        # Viseme groups based on articulation and mouth shape
        phoneme_groups = {
            1: ["m", "b", "p"],          # Bilabials (lips together)
            2: ["f", "v"],               # Labiodentals (lip and teeth)
            3: ["th", "dh"],             # Dentals (tongue near teeth)
            4: ["t", "d", "s", "z", "n", "l", "dx", "r"],  # Alveolars (tongue near alveolar ridge)
            5: ["sh", "zh"],             # Postalveolars (tongue near hard palate)
            6: ["j", "ch", "jh", "y"],   # Palatals (tongue near palate)
            7: ["k", "g", "ng", "w"],    # Velars (tongue near velum, includes "w" for labio-velar)
            8: ["hh", "h#"],             # Glottals (throat, back of mouth)
            9: ["aa", "ae", "ah", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"],  # Vowels
            10: ["|"],                   # Space (Pause between words)
            11: ["spn"],                 # Silence (spn)
            12: ["[UNK]"],               # Unknown phoneme
            13: ["[PAD]"],               # Padding token
            14: ["<s>"],                 # Start of sequence
            15: ["</s>"],                # End of sequence
        }

        id_to_group = {}
        def get_phoneme_group_number(phoneme):
            # Determine the viseme group for each phoneme
            for group_number, phonemes in phoneme_groups.items():
                if phoneme in phonemes:
                    return group_number
            return -1

        processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
        vocab_size = processor.tokenizer.vocab_size

        # Map each phoneme ID to its corresponding viseme group
        for idx in range(vocab_size):
            phoneme = processor.tokenizer.convert_ids_to_tokens([idx])[0]
            phoneme_group_number = get_phoneme_group_number(phoneme)
            id_to_group[idx] = phoneme_group_number

        return id_to_group

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
            # Apply replayed transformation when test is False
            transformed = A.ReplayCompose.replay(
                repeat_transform,
                image=cropped_image,
                mask=1 - hull_mask,
                keypoints=cropped_landmarks_fan,
                mediapipe_keypoints=cropped_landmarks_mediapipe
            )
        else:
            # Apply resize transformation when test is True
            transformed = resize(
                image=cropped_image,
                mask=1 - hull_mask,
                keypoints=cropped_landmarks_fan,
                mediapipe_keypoints=cropped_landmarks_mediapipe
            )

        return index, transformed
    
    def drop_audio_segments(self, video_dict):
        """
        Drops audio segments that correspond to dropped frames in the video.
        """
        dropped_frames = video_dict.get("dropped_frames", torch.tensor([], dtype=torch.int32)).to(video_dict["img"].device)
        audio_sample_rate = video_dict["audio_sample_rate"].item()
        video_frame_rate = video_dict["fps"].item()

        # Calculate total video frames (including dropped)
        total_video_frames = video_dict["img"].shape[0] + dropped_frames.shape[0]

        # Calculate timestamps for each frame
        frame_timestamps = torch.linspace(0, total_video_frames / video_frame_rate, total_video_frames, device=dropped_frames.device)

        # Remove dropped frame timestamps
        valid_timestamps = torch.cat([frame_timestamps[:dropped_frames[0]], frame_timestamps[dropped_frames[-1] + 1:]]) if dropped_frames.numel() > 0 else frame_timestamps

        # Calculate duration of each valid video frame in seconds
        frame_duration = 1.0 / video_frame_rate

        # Calculate corresponding audio indices for each valid frame timestamp
        audio_start_indices = (valid_timestamps * audio_sample_rate).long()
        audio_end_indices = ((valid_timestamps + frame_duration) * audio_sample_rate).long()

        # Create a new audio tensor by combining the valid audio segments
        audio_segments = [video_dict["audio"][start:end] for start, end in zip(audio_start_indices, audio_end_indices)]
        video_dict["audio"] = torch.cat(audio_segments)
        return video_dict

    def create_phoneme_group_timestamps(self, video_dict):
        """
        Creates group timestamps given audio, video frames, dropped frames, and phoneme group mappings.
        Output: (GROUP ID, phoneme, start_idx, end_idx) where start_idx and end_idx are indices in img_tensor.
        """
        img_tensor = video_dict["img"]
        dropped_frames = video_dict.get("dropped_frames", torch.tensor([], dtype=torch.int32)).to(img_tensor.device)
        sequence_tensor = video_dict["audio_phonemes"][0]
        audio_sample_rate = video_dict["audio_sample_rate"]  # Accessing attributes directly
        video_frame_rate = video_dict["fps"]  # Accessing attributes directly

        # Step 1: Measure the length of the audio in seconds
        audio_length_seconds = len(img_tensor) / video_frame_rate

        # Step 2: Calculate the number of video frames based on img_tensor shape and dropped frames
        total_frames = img_tensor.shape[0] + dropped_frames.shape[0]

        # Step 3: Calculate the timestamp for each video frame (including dropped frames)
        frame_timestamps = torch.linspace(0, total_frames / video_frame_rate, total_frames, device=img_tensor.device)

        # Step 5: Calculate phoneme durations in seconds
        phoneme_duration = audio_length_seconds / len(sequence_tensor)

        # Step 6: Create the array of tuples (group, phoneme, start_time, end_time) for valid groups (1-9)
        phoneme_id_to_group = self.phoneme_map
        group_timestamps = []

        current_group = -1
        group_start_time = None
        group_end_time = None
        current_phoneme = None  # Store the current phoneme

        for i, phoneme_id in enumerate(sequence_tensor):
            group = phoneme_id_to_group.get(int(phoneme_id.item()), -1)

            if group == 13:  # Handle "[PAD]" (group 13) by extending the current group's end time
                if current_group != -1 and group_start_time is not None:
                    group_end_time += phoneme_duration
                continue  # Do not change the current group

            if group == -1 or group > 9:  # End the current group for irrelevant groups
                if current_group != -1 and group_start_time is not None:
                    # Close the ongoing group and append to group_timestamps
                    group_timestamps.append((current_group, current_phoneme, group_start_time, group_end_time))
                    current_group = -1  # Reset the current group
                    group_start_time = None
                    group_end_time = None
                    current_phoneme = None
                continue  # Move to the next phoneme

            # If we reach here, it's a valid group (1-9)
            if current_group == -1:
                current_group = group
                current_phoneme = phoneme_id.item()  # Track the original phoneme
                group_start_time = i * phoneme_duration
                group_end_time = group_start_time + phoneme_duration
            elif current_group != group:
                # Close the previous group and start a new group
                group_timestamps.append((current_group, current_phoneme, group_start_time, group_end_time))

                current_group = group
                current_phoneme = phoneme_id.item()  # Update to the new phoneme
                group_start_time = group_end_time
                group_end_time = group_start_time + phoneme_duration
            else:
                # Continue extending the current group's end time
                group_end_time += phoneme_duration

        # Ensure any open segment is closed at the end of the sequence
        if current_group != -1 and group_start_time is not None:
            group_timestamps.append((current_group, current_phoneme, group_start_time, audio_length_seconds))

        # Step 7: Identify contiguous sequences of dropped frames
        if dropped_frames.numel() > 0:
            dropped_sequences = []
            start_idx = dropped_frames[0].item()
            for i in range(1, len(dropped_frames)):
                if dropped_frames[i] != dropped_frames[i - 1] + 1:
                    dropped_sequences.append((start_idx, dropped_frames[i - 1].item()))
                    start_idx = dropped_frames[i].item()
            dropped_sequences.append((start_idx, dropped_frames[-1].item()))

            # Step 7.1: Adjust group timestamps based on dropped frame sequences and remove fully dropped groups
            adjusted_group_timestamps = []
            for group, phoneme, start_time, end_time in group_timestamps:
                fully_dropped = False
                for drop_start, drop_end in dropped_sequences:
                    drop_start_time = frame_timestamps[drop_start].item()
                    drop_end_time = frame_timestamps[drop_end].item()

                    # Check if the dropped sequence completely overlaps the group (fully dropped group)
                    if drop_start_time <= start_time and drop_end_time >= end_time:
                        fully_dropped = True
                        break

                    # Adjust if the dropped sequence overlaps with the start of the group
                    if drop_start_time <= start_time <= drop_end_time:
                        start_time = min(drop_end_time + (1.0 / video_frame_rate), audio_length_seconds)

                    # Adjust if the dropped sequence overlaps with the end of the group
                    if drop_start_time <= end_time <= drop_end_time:
                        end_time = max(drop_start_time - (1.0 / video_frame_rate), 0)

                    # If the dropped sequence is in-between, reduce the group length accordingly
                    if start_time < drop_start_time < end_time:
                        duration_reduction = (drop_end_time - drop_start_time) + (1.0 / video_frame_rate)
                        end_time -= duration_reduction

                if not fully_dropped:
                    # Append the adjusted group timestamp only if it wasn't fully dropped
                    adjusted_group_timestamps.append((group, phoneme, start_time, end_time))
                    
            group_timestamps = adjusted_group_timestamps

        # Step 9: Convert adjusted timestamps into img_tensor indices
        video_group_indices = []
        for group, phoneme, start_time, end_time in group_timestamps:
            start_idx = math.floor(start_time * video_frame_rate)  # Convert start time to index
            end_idx = math.ceil(end_time * video_frame_rate)      # Convert end time to index

            # Ensure indices are within img_tensor bounds
            start_idx = max(0, min(start_idx, img_tensor.shape[0] - 1))
            end_idx = max(0, min(end_idx, img_tensor.shape[0] - 1))

            if start_idx == end_idx:
                continue
            video_group_indices.append((group, phoneme, start_idx, end_idx))

        # # Step 10: Finally, filter out false-positives using the text-based phonemes
        # text_phoneme_groups = [phoneme_id_to_group.get(int(x.item()), -1) for x in text_tensor]
        # final_group_indices = []
        # for group in text_phoneme_groups:
        #     for i, group_idx in enumerate(video_group_indices):
        #         if group_idx[0] == group:
        #             final_group_indices.append(group_idx)
        #             video_group_indices = video_group_indices[min(i + 1, len(video_group_indices)):]
        #             break
        
        return video_group_indices

    def phoneme_group_agreement(self, video_dict, phoneme_map):
        """
        Measures the agreement in present phoneme groups between audio and text phonemes.
        :param video_dict: Dictionary containing "audio_phonemes" and "text_phonemes"
        :param phoneme_map: Dictionary mapping phoneme IDs to their groups (0-9)
        :return: Percentage of agreement between audio and text phoneme groups
        """
        audio_phonemes = video_dict["audio_phonemes"][0]
        text_phonemes = video_dict["text_phonemes"]

        # Map phonemes to their groups for audio and text
        audio_groups = {phoneme_map[phoneme.item()] for phoneme in audio_phonemes if phoneme_map[phoneme.item()] in range(0, 10)}
        text_groups = {phoneme_map[phoneme.item()] for phoneme in text_phonemes if phoneme_map[phoneme.item()] in range(0, 10)}

        # Measure overlap (intersection) between audio and text phoneme groups
        common_groups = audio_groups.intersection(text_groups)

        # Quantify the overlap
        if len(text_groups) == 0:
            return 0.0  # Avoid division by zero if there are no groups
        overlap_percentage = len(common_groups) / len(text_groups)

        return overlap_percentage

    def __getitem_aux__(self, index):
        v, s, start, end = self.split_idxs[index]
        v = int(v)
        s = int(s)

        start = int(start)
        end = int(end)
        rand_int = random.randint(start, end - 2)

        start = rand_int
        end = rand_int + 1

        video_group = self._get_hdf5_file(int(v))[str(s)]
        video_dict = {}
        video_dict["fps"] = video_group.attrs["fps"]
        video_dict["audio_sample_rate"] = video_group.attrs["sample_rate"]

        # Gather all image data (subsample if necessary)
        for key, item in video_group.items():
            value = item[()]
            if isinstance(value, np.ndarray) and key in self.framewise_keys:
                sampled_value = value.copy()  # Copy the data to avoid issues with multiprocessing
                sampled_value = sampled_value[start:end]
                video_dict[key] = sampled_value
                # print(key, sampled_value.shape)
        
        video_dict["audio_phonemes"] = video_group["audio_phonemes"][()].copy()
        video_dict["text_phonemes"] = video_group["text_phonemes"][()].copy()

        wav2vec_video_group = self._get_wav2vec_hdf5_file(int(v))[str(s)]
        video_dict["wav2vec_frame_level_embeddings"] = wav2vec_video_group["wav2vec_frame_level_embeddings"][()].copy()
        video_dict["wav2vec_frame_level_embeddings"] = video_dict["wav2vec_frame_level_embeddings"][start:end]

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
        else:
            repeat_transform = None

        # Properly pass all arguments for multiprocessing
        args_list = [(i, video_dict["img"][i], video_dict["mask"][i], video_dict["landmarks_fan"][i],
                      video_dict["landmarks_mp"][i], repeat_transform, self.test, self.resize) 
                      for i in range(len(video_dict["img"]))]

        # Use apply_async with a timeout and maintain order
        results = []
        for args in args_list:
            results.append(self.process_frame(args))

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
        video_dict["img_mica"] = video_dict["img_mica"].astype(np.float32)

        # Turning numpy arrays into tensors
        for key, item in video_dict.items():
            if isinstance(item, np.ndarray) and item.dtype != "S1":
                video_dict[key] = torch.tensor(item)
        
        # If missing data, mark it as audio/no audio
        if "audio_phonemes" not in video_dict.keys():
            video_dict["audio_phonemes"] = []
            video_dict["text_phonemes"] = []
            video_dict["silent"] = torch.tensor([True])
            video_dict["phoneme_timestamps"] = []
        elif len(["audio_phonemes"]) == 0:
            video_dict["silent"] = torch.tensor([True])
            video_dict["phoneme_timestamps"] = []
        else:
            video_dict["silent"] = torch.tensor([False])
            video_dict["phoneme_timestamps"] = self.create_phoneme_group_timestamps(video_dict)
            # video_dict = self.drop_audio_segments(video_dict)

            video_dict["phoneme_timestamps"] = [(x, y, s, e) for (x, y, s, e) in video_dict["phoneme_timestamps"] if s >= start and e <= end]
            video_dict["phoneme_timestamps"] = [(x, y, s - start, e - start) for (x, y, s, e) in video_dict["phoneme_timestamps"]]
            if len(video_dict["phoneme_timestamps"]) == 0:
                video_dict["silent"] = torch.tensor([True])

        if "text" in video_dict.keys():
            del video_dict["text"]
            del video_dict["text_phonemes"]
        
        if "wav2vec_frame_level_embeddings" not in video_dict.keys():
            print("No Audio Embeddings")
            raise Exception
        else:
            video_dict["audio_feat"] = video_dict["wav2vec_frame_level_embeddings"]

        return video_dict

def slice_with_overlap(img_tensor, max_batch_len, overlap=3):
    """
    Slices an image tensor (M, H, W, 3) into multiple overlapping slices.

    Args:
        img_tensor (numpy.ndarray): The input image tensor of shape (M, H, W, 3).
        max_batch_len (int): The maximum number of frames per slice.
        overlap (int): The number of overlapping frames (default is 3).

    Returns:
        list: A list of slices from the image tensor.
    """
    img_length = img_tensor.shape[0]
    
    # Calculate the number of slices needed
    num_slices = math.ceil((img_length - overlap) / (max_batch_len - overlap))

    # Initialize start and end indices
    idx_start = np.arange(num_slices) * (max_batch_len - overlap)
    idx_end = idx_start + max_batch_len

    # Ensure the last slice does not exceed the image length
    idx_end = np.clip(idx_end, None, img_length)

    # Adjust the start indices for the last slice if needed
    idx_start = np.clip(idx_start, None, idx_end - max_batch_len)

    # Collect the slices
    slices = [img_tensor[s:e] for s, e in zip(idx_start, idx_end)]

    return slices

def cull_indices(hdf5_file_paths, seg_indices, config, heuristic_probs=(0/12, 6/12, 6/12), length_bias=1.0, variance_bias=10.0, slice_properties_path="slice_properties.pkl"):
    """
    Filters and generates valid indices for data slices from HDF5 files with a maximum cap,
    using a 2D Gaussian sampling strategy to favor long slices with high variance.

    Args:
        hdf5_file_paths (list): List of paths to HDF5 files.
        seg_indices (numpy.ndarray): Array of (file_idx, group_idx) pairs to be processed.
        config: Configuration object with dataset thresholds and max cap.
        heuristic_probs (tuple): Probabilities for the three heuristics (random, mouth variance, overall variance).
        bias_factor (float): Factor to scale the covariance matrix bias, favoring long slices with high variance.
        slice_properties_path (str): Path to save/load precomputed slice properties.

    Returns:
        numpy.ndarray: Array of valid indices [file_idx, group_idx, start_idx, end_idx].
    """
    data_idxs = []
    max_cap = config.dataset.iHiTOP.max_data_idxs
    seg_indices_by_file = {}

    # Landmark indices for mouth variance calculation
    mp_upper_inner_lip_idxs = [66, 73, 74, 75, 76, 86, 91, 92, 93, 94, 104]
    mp_lower_inner_lip_idxs = [67, 78, 79, 81, 83, 96, 97, 99, 101]

    sampled_randomly = []
    sampled_segments = []
    fallback_count = 0

    # Group segment indices by file
    for file_idx, group_idx in seg_indices:
        if file_idx not in seg_indices_by_file:
            seg_indices_by_file[file_idx] = []
        seg_indices_by_file[file_idx].append(int(group_idx))

    if os.path.exists(slice_properties_path):
        with open(slice_properties_path, "rb") as f:
            slice_properties = pickle.load(f)
        print(f"Loaded precomputed slice properties from {slice_properties_path}")
    else:
        # Compute slice properties and save them for reuse
        hdf5_files = []
        slice_properties = {}
        for file_idx, file_path in tqdm(enumerate(hdf5_file_paths), desc="Processing HDF5 Files"):
            hdf5_file = h5py.File(file_path, "r")
            hdf5_files.append(hdf5_file)

            slices_info = []

            for group_idx in seg_indices_by_file.get(file_idx, []):
                group = hdf5_file[str(group_idx)]

                try:
                    required_keys = ["img", "landmarks_mp", "audio_phonemes", "hubert_frame_level_embeddings_v4"]
                    if not all(key in group.keys() for key in required_keys):
                        continue

                    img_length = len(group["img"])
                    removed_frames_length = len(group["removed_frames"])
                    total_length = img_length + removed_frames_length
                    removed_frames_percentage = removed_frames_length / total_length if total_length > 0 else 0

                    if removed_frames_percentage >= config.dataset.iHiTOP.removed_frames_threshold:
                        continue
                    if img_length > config.dataset.iHiTOP.max_seg_len or img_length < config.dataset.iHiTOP.min_seg_len:
                        continue

                    # Generate slice indices
                    idx_start = np.arange(math.ceil((img_length - config.train.split_overlap) /
                                                    (config.train.max_batch_len - config.train.split_overlap)))
                    idx_end = idx_start + 1
                    idx_start *= (config.train.max_batch_len - config.train.split_overlap)
                    idx_end *= (config.train.max_batch_len - config.train.split_overlap)
                    idx_end += config.train.split_overlap
                    idx_end = np.clip(idx_end, None, img_length)

                    for s, e in zip(idx_start, idx_end):
                        slice_length = e - s
                        landmarks = group["landmarks_mp"][s:e]
                        overall_variance = np.var(landmarks, axis=(0, 1)).sum()
                        lip_landmarks = np.concatenate([
                            landmarks[:, mp_upper_inner_lip_idxs, :],
                            landmarks[:, mp_lower_inner_lip_idxs, :]
                        ], axis=1)
                        mouth_variance = np.var(lip_landmarks, axis=(0, 1)).sum()
                        slices_info.append((slice_length, overall_variance, mouth_variance, file_idx, group_idx, s, e))
                except Exception as e:
                    continue

            slice_properties[file_idx] = slices_info

        # Save slice properties
        with open(slice_properties_path, "wb") as f:
            pickle.dump(slice_properties, f)
        print(f"Saved slice properties to {slice_properties_path}")

        for hdf5_file in hdf5_files:
            if hdf5_file is not None:
                hdf5_file.close()

    # Sampling using heuristics
    remaining_cap = max_cap
    file_segment_counts = {}  # Track number of segments sampled per file

    with tqdm(total=max_cap, desc="Sampling Segments") as pbar:
        while remaining_cap > 0 and any(slice_properties.values()):
            # Randomly shuffle files to sample from
            file_indices = list(slice_properties.keys())
            np.random.shuffle(file_indices)

            for file_idx in file_indices:
                if not slice_properties[file_idx]:
                    continue

                # Extract slice properties
                slice_array = np.array([(length, overall_var, mouth_var, group_idx, s, e) for length, overall_var, mouth_var, _, group_idx, s, e in slice_properties[file_idx]])
                slice_lengths, overall_variances, mouth_variances = slice_array[:, 0], slice_array[:, 1], slice_array[:, 2]

                # Define bins for segment lengths
                bin_edges = np.arange(1, 62, 1)  # Bins from 1 to 60
                bins = np.digitize(slice_lengths, bin_edges)

                # Group slices into bins without initial sorting
                binned_slices = {i: [] for i in range(1, len(bin_edges))}
                for idx, bin_idx in enumerate(bins):
                    if 1 <= bin_idx < len(bin_edges):  # Ignore out-of-range bins
                        binned_slices[bin_idx].append((slice_lengths[idx], overall_variances[idx], mouth_variances[idx], slice_array[idx, 3], slice_array[idx, 4], slice_array[idx, 5]))

                # Define a skewed normal distribution for bin sampling
                desired_mean = 40
                skewness = 0.7
                scale = 25
                skew_dist = skewnorm(skewness, loc=desired_mean, scale=scale)

                # Generate probabilities for bins based on bin centers
                bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])
                valid_bins = [i for i in binned_slices if binned_slices[i]]  # Only consider non-empty bins

                if not valid_bins:  # Skip if no valid bins
                    slice_properties.pop(file_idx)
                    continue

                valid_bin_centers = bin_centers[np.array(valid_bins) - 1]  # Adjust indices to align with `bin_centers`

                # Adjust the mean if longer bins are empty
                adjusted_mean = min(desired_mean + 5, valid_bin_centers.max())  # Push toward max bin center
                skew_dist = skewnorm(skewness, loc=adjusted_mean, scale=scale)

                bin_probs = skew_dist.pdf(valid_bin_centers)
                bin_probs /= bin_probs.sum()  # Normalize to sum to 1

                # Sample a single bin asymmetrically
                sampled_bin = np.random.choice(valid_bins, p=bin_probs)

                # Sample a slice from the selected bin
                if binned_slices[sampled_bin]:
                    heuristic = np.random.rand()
                    if heuristic < heuristic_probs[0]:  # Random sampling
                        slice_idx = np.random.randint(len(binned_slices[sampled_bin]))
                    elif heuristic < heuristic_probs[0] + heuristic_probs[1]:  # Mouth variance heuristic
                        # Sort the bin by mouth variance before sampling
                        binned_slices[sampled_bin].sort(key=lambda x: x[2], reverse=True)
                        slice_idx = 0  # Select the highest mouth variance
                    else:  # Overall variance heuristic
                        # Sort the bin by overall variance before sampling
                        binned_slices[sampled_bin].sort(key=lambda x: x[1], reverse=True)
                        slice_idx = 0  # Select the highest overall variance

                    sampled_slice = binned_slices[sampled_bin].pop(slice_idx)
                    length, overall_var, mouth_var, group_idx, s, e = sampled_slice

                    # Add to sampled results
                    sampled_segments.append(sampled_slice)
                    file_segment_counts[file_idx] = file_segment_counts.get(file_idx, 0) + 1
                    data_idxs.append(np.array([file_idx, group_idx, s, e]))
                    remaining_cap -= 1
                    pbar.update(1)

                    # Stop if the max cap is reached
                    if remaining_cap <= 0:
                        break

                # If bins are exhausted for this file, remove it from consideration
                if all(not binned_slices[bin_idx] for bin_idx in binned_slices):
                    slice_properties.pop(file_idx)

    # Gather all slices and sampled slices
    all_slices = []
    for slices in slice_properties.values():
        all_slices.extend([(s[0], s[1], s[2]) for s in slices])  # (length, overall_var, mouth_var)
    all_slices = np.array(all_slices)

    sampled_slices = np.array([(s[0], s[1], s[2]) for s in sampled_segments])

    # Initialize categories to avoid UnboundLocalError
    not_sampled = []
    sampled = []

    # Categorize slices based on sampling status
    for slice_data in all_slices:
        if tuple(slice_data) in sampled_slices:
            sampled.append(slice_data)
        else:
            not_sampled.append(slice_data)

    # Convert lists to numpy arrays for consistent processing
    not_sampled = np.array(not_sampled)
    sampled = np.array(sampled)

    # Ensure there are valid slices to visualize
    if len(all_slices) == 0 or len(sampled_slices) == 0:
        print("No slices or sampled slices to visualize.")
    else:
        total_points = 10000
        total_available = len(not_sampled) + len(sampled)

        if total_available > total_points:
            not_sampled_limit = int((len(not_sampled) / total_available) * total_points)
            sampled_limit = total_points - not_sampled_limit

            # Downsample based on proportional limits
            if len(not_sampled) > not_sampled_limit:
                not_sampled = not_sampled[np.random.choice(len(not_sampled), not_sampled_limit, replace=False)]
            if len(sampled) > sampled_limit:
                sampled = sampled[np.random.choice(len(sampled), sampled_limit, replace=False)]

        # Scatter plot for overall variance
        plt.figure(figsize=(10, 6))

        # Plot downsampled categories
        if len(not_sampled) > 0:
            plt.scatter(not_sampled[:, 0], not_sampled[:, 1], color='gray', label='Not Sampled', s=5)
        if len(sampled) > 0:
            plt.scatter(sampled[:, 0], sampled[:, 1], color='red', label='Sampled', s=10)

        # Add labels, title, and legend
        plt.title("Sampling Density: Slice Length vs. Overall Variance")
        plt.xlabel("Slice Length")
        plt.ylabel("Overall Variance")
        plt.legend()
        plt.savefig("scatter_sampling_density_overall_variance.png")
        plt.close()

        # Scatter plot for mouth variance
        plt.figure(figsize=(10, 6))

        # Plot downsampled categories
        if len(not_sampled) > 0:
            plt.scatter(not_sampled[:, 0], not_sampled[:, 2], color='gray', label='Not Sampled', s=5)
        if len(sampled) > 0:
            plt.scatter(sampled[:, 0], sampled[:, 2], color='red', label='Sampled', s=10)

        # Add labels, title, and legend
        plt.title("Sampling Density: Slice Length vs. Mouth Variance")
        plt.xlabel("Slice Length")
        plt.ylabel("Mouth Variance")
        plt.legend()
        plt.savefig("scatter_sampling_density_mouth_variance.png")
        plt.close()

        # Statistics
        avg_all_length = np.mean(all_slices[:, 0])
        avg_sampled_length = np.mean(sampled_slices[:, 0])

        avg_all_overall_var = np.mean(all_slices[:, 1])
        avg_sampled_overall_var = np.mean(sampled_slices[:, 1])

        avg_all_mouth_var = np.mean(all_slices[:, 2])
        avg_sampled_mouth_var = np.mean(sampled_slices[:, 2])

        max_length_all = np.max(all_slices[:, 0])
        max_length_sampled = np.max(sampled_slices[:, 0])

        min_length_all = np.min(all_slices[:, 0])
        min_length_sampled = np.min(sampled_slices[:, 0])

        print("Statistics:")
        print(f"  Average Length (All): {avg_all_length:.2f}")
        print(f"  Average Length (Sampled): {avg_sampled_length:.2f}")
        print(f"  Max Length (All): {max_length_all:.2f}")
        print(f"  Max Length (Sampled): {max_length_sampled:.2f}")
        print(f"  Min Length (All): {min_length_all:.2f}")
        print(f"  Min Length (Sampled): {min_length_sampled:.2f}")

        print(f"  Average Overall Variance (All): {avg_all_overall_var:.4f}")
        print(f"  Average Overall Variance (Sampled): {avg_sampled_overall_var:.4f}")
        print(f"  Average Mouth Variance (All): {avg_all_mouth_var:.4f}")
        print(f"  Average Mouth Variance (Sampled): {avg_sampled_mouth_var:.4f}")

        print(f"  Fallbacks Due to SVD Failure: {fallback_count}")
        print(f"  Unique Files Sampled: {len(file_segment_counts)}")
        print(f"  Average Segments Per File: {np.mean(list(file_segment_counts.values())):.2f}")
        print(f"  Max Segments From a File: {np.max(list(file_segment_counts.values()))}")
        print(f"  Min Segments From a File: {np.min(list(file_segment_counts.values()))}")

        print("Plots saved as:")
        print("- scatter_sampling_density_overall_variance.png")
        print("- scatter_sampling_density_mouth_variance.png")

    return np.array(data_idxs)

def cull_indices_old(hdf5_file_paths, seg_indices, config):
    data_idxs = []

    # Open all HDF5 files before the loop
    hdf5_files = []
    try:
        for file_path in hdf5_file_paths:
            hdf5_files.append(h5py.File(file_path, "r"))

        # Loop over the seg_indices and check against the threshold
        for i, (file_idx, group_idx) in enumerate(seg_indices):
            try:
                file_idx = int(file_idx)
                group_idx = str(int(group_idx))  # Convert group index to string

                hdf5_file = hdf5_files[file_idx]
                group = hdf5_file[group_idx]

                if "hubert_frame_level_embeddings_v4" not in group.keys():
                    continue

                img_length = len(group["img"])
                removed_frames_length = len(group["removed_frames"])
                total_length = img_length + removed_frames_length

                # Calculate removed frames percentage
                if total_length > 0:  # Avoid division by zero
                    removed_frames_percentage = removed_frames_length / total_length
                else:
                    removed_frames_percentage = 0

                # If the percentage exceeds the threshold, mark the index as invalid
                if removed_frames_percentage >= config.dataset.iHiTOP.removed_frames_threshold:
                    continue

                # Also, remove if img_length is too long
                if img_length > config.dataset.iHiTOP.max_seg_len:
                    continue

                # Or too short
                if img_length < config.dataset.iHiTOP.min_seg_len:
                    continue

                # Record the start and end indices associated with each subsegment
                idx_start = np.arange(math.ceil((img_length - config.train.split_overlap) / (config.train.max_batch_len - config.train.split_overlap)))
                idx_end = idx_start + 1

                idx_start *= (config.train.max_batch_len - config.train.split_overlap)
                idx_end *= (config.train.max_batch_len - config.train.split_overlap)
                idx_end += config.train.split_overlap

                # Clip the end indices to not exceed img_length
                idx_end = np.clip(idx_end, None, img_length)

                for s, e in zip(idx_start, idx_end):
                    data_idxs.append(np.array([file_idx, group_idx, s, e])[None])
            except Exception as e:
                print(f"Segment {group_idx} failed from {file_idx}!")
                print(e)
                continue
    finally:
        # Close all HDF5 files
        for hdf5_file in hdf5_files:
            hdf5_file.close()

    return np.concatenate(data_idxs)

def get_datasets_iHiTOP_parallel(config=None):
    if get_rank() == 0:
        # Rank 0 initializes files and indices
        hdf5_file_paths = []
        bad_files = []

        for file in os.listdir(config.dataset.iHiTOP.hdf5_path):
            try:
                file_path = os.path.join(config.dataset.iHiTOP.hdf5_path, file)
                # Validate files
                with h5py.File(file_path, 'r') as hdf5_file:
                    pass  # Just verify the file can be opened
                hdf5_file_paths.append(file_path)
            except:
                print(f"Bad File {file}")
                bad_files.append(file)

        np.save(config.dataset.iHiTOP.bad_files, np.array(bad_files))

        if not os.path.exists(config.dataset.iHiTOP.data_idxs):
            print("Data indices undefined! Defining now....")
            seg_indices = []

            for i, hdf5_file in enumerate(hdf5_file_paths):
                with h5py.File(hdf5_file, 'r') as h5:
                    for key in h5.keys():
                        seg_indices.append(np.array([i, int(key)])[None])
            seg_indices = np.concatenate(seg_indices)

            data_idxs = cull_indices(hdf5_file_paths, seg_indices, config)
            np.save(config.dataset.iHiTOP.data_idxs, data_idxs)

        if not os.path.exists(config.dataset.iHiTOP.final_idxs_train):
            data_idxs = np.load(config.dataset.iHiTOP.data_idxs)

            # Group data_idxs by file_idx
            file_groups = {}
            for idx in data_idxs:
                file_idx = idx[0]
                if file_idx not in file_groups:
                    file_groups[file_idx] = []
                file_groups[file_idx].append(idx)

            # Shuffle file indices
            file_indices = list(file_groups.keys())
            np.random.shuffle(file_indices)

            # Assign files to train, val, and test sets
            train_set, val_set, test_set = [], [], []
            total_segments = len(data_idxs)
            train_threshold = int(config.dataset.iHiTOP.train_percentage * total_segments)
            val_threshold = int(config.dataset.iHiTOP.val_percentage * total_segments)

            segment_count = 0
            for file_idx in file_indices:
                if segment_count < train_threshold:
                    train_set.extend(file_groups[file_idx])
                elif segment_count < train_threshold + val_threshold:
                    val_set.extend(file_groups[file_idx])
                else:
                    test_set.extend(file_groups[file_idx])

                segment_count += len(file_groups[file_idx])

            train_set = np.array(train_set)
            val_set = np.array(val_set)
            test_set = np.array(test_set)

            np.save(config.dataset.iHiTOP.final_idxs_train, train_set)
            np.save(config.dataset.iHiTOP.final_idxs_val, val_set)
            np.save(config.dataset.iHiTOP.final_idxs_test, test_set)
            print("Shuffled indices defined!")
    else:
        # Other ranks wait until rank 0 finishes
        barrier()

    # All ranks proceed after synchronization
    hdf5_file_paths = [os.path.join(config.dataset.iHiTOP.hdf5_path, file) for file in os.listdir(config.dataset.iHiTOP.hdf5_path)]
    train_idxs = np.load(config.dataset.iHiTOP.final_idxs_train)
    
    val_idxs = np.load(config.dataset.iHiTOP.final_idxs_val)
    # rand_idx = np.random.choice(len(val_idxs), size=3000, replace=False)
    # val_idxs = val_idxs[rand_idx]
    test_idxs = np.load(config.dataset.iHiTOP.final_idxs_test)

    return (
        iHiTOPDatasetParallel(hdf5_file_paths, train_idxs, config, test=False),
        iHiTOPDatasetParallel(hdf5_file_paths, val_idxs, config, test=True),
        iHiTOPDatasetParallel(hdf5_file_paths, test_idxs, config, test=True),
        len(train_idxs) + len(val_idxs) + len(test_idxs)
    )
