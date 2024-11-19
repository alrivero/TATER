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


class iHiTOPDataset(BaseVideoDataset):
    def __init__(self, video_data, split_idxs, config, test=False):
        super().__init__(video_data, split_idxs, config, test)
        self.name = 'iHiTOP'
        self.iHiTOP_frame_step = config.dataset.iHiTOP.frame_step

        self.phoneme_map = self.create_phoneme_map()

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
        audio_tensor = video_dict["audio"]
        img_tensor = video_dict["img"]
        dropped_frames = video_dict.get("dropped_frames", torch.tensor([], dtype=torch.int32)).to(img_tensor.device)
        sequence_tensor = video_dict["audio_phonemes"][0]
        text_tensor = video_dict["text_phonemes"]
        audio_sample_rate = video_dict["audio_sample_rate"]  # Accessing attributes directly
        video_frame_rate = video_dict["fps"]  # Accessing attributes directly

        # Step 1: Measure the length of the audio in seconds
        audio_length_seconds = audio_tensor.size(0) / audio_sample_rate

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

        # Step 10: Finally, filter out false-positives using the text-based phonemes
        text_phoneme_groups = [phoneme_id_to_group.get(int(x.item()), -1) for x in text_tensor]
        final_group_indices = []
        for group in text_phoneme_groups:
            for i, group_idx in enumerate(video_group_indices):
                if group_idx[0] == group:
                    final_group_indices.append(group_idx)
                    video_group_indices = video_group_indices[min(i + 1, len(video_group_indices)):]
                    break
        
        return final_group_indices


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

        video_dict["fps"] = video_group.attrs["fps"]
        video_dict["audio_sample_rate"] = video_group.attrs["sample_rate"]

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
        
        # with Pool(self.num_workers) as pool:
        #     async_results = [pool.apply_async(self.process_frame, args=(arg,)) for arg in args_list]
            
        #     for res in async_results:
        #         try:
        #             # Set a timeout of 30 seconds for each task
        #             result = res.get(timeout=30)
        #             results.append(result)
        #         except TimeoutError:
        #             print(f"{index} timed out and was skipped.")
        #             continue

        # # Sort results by the original index to maintain the order
        # results = sorted(results, key=lambda x: x[0])

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
            video_dict = self.drop_audio_segments(video_dict)

        return video_dict

def cull_indices(hdf5_files, seg_indices, config):
    valid_mask = np.ones(len(seg_indices), dtype=bool)  # Start with all indices valid
    effective_seg_count = 0

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
            if removed_frames_percentage >= config.dataset.iHiTOP.removed_frames_threshold:
                valid_mask[i] = False  # Mark invalid if threshold is exceeded

            # Also, remove if img_length is too long
            if img_length > config.dataset.iHiTOP.max_seg_len:
                valid_mask[i] = False 

            # Or too short
            if img_length < config.dataset.iHiTOP.min_seg_len:
                valid_mask[i] = False 

            # Record effective seg count
            if valid_mask[i]:
                effective_seg_count += math.ceil(img_length / config.train.max_batch_len)
        except:
            print(f"Segment {group_idx} failed from {file_idx}!")
            valid_mask[i] = False
    
    return effective_seg_count, valid_mask  # Return the boolean mask

def get_datasets_iHiTOP(config=None):
    # Gather all hdf5 files and number of segments in each
    hdf5_files = []
    seg_counts = []
    bad_files = []
    for file in os.listdir(config.dataset.iHiTOP.hdf5_path):
        hdf5_file = None
        try:
            hdf5_file = h5py.File(os.path.join(
                config.dataset.iHiTOP.hdf5_path,
                file
            ), 'r')
            seg_count = len(hdf5_file.keys())

            hdf5_files.append(hdf5_file)
            seg_counts.append(seg_count)
            break
        except:
            print(f"Bad File {file}")
            bad_files.append(file)
            if hdf5_file is not None:
                hdf5_file.close()
    
    if not os.path.exists(config.dataset.iHiTOP.bad_files):
        np.save(config.dataset.iHiTOP.bad_files, np.array(bad_files))

    seg_indices = []
    # Set up indexing to access all of these files
    for i, hdf5_file in enumerate(hdf5_files):
        for key in hdf5_file.keys():
            seg_indices.append(np.array([i, int(key)])[None])
    seg_indices = np.concatenate(seg_indices)

    # Now cull indices depending on their number of removed frames
    if not os.path.exists(config.dataset.iHiTOP.data_idxs):
        print("Data indices undefined! Defining now....")

        effective_seg_count, data_idxs = cull_indices(hdf5_files, seg_indices, config)

        np.save(config.dataset.iHiTOP.data_idxs, data_idxs)
        np.save(config.dataset.iHiTOP.effective_seg_count, effective_seg_count)
        print("Data indices defined!")

    data_idxs = np.load(config.dataset.iHiTOP.data_idxs)
    effective_seg_count = np.load(config.dataset.iHiTOP.effective_seg_count)
    seg_indices = seg_indices[data_idxs]
    print(f"Effective Segment count {effective_seg_count}")
    print(f"Actual Segment count {len(seg_indices)}")


    # Assert train, val, test split
    assert config.dataset.iHiTOP.train_percentage + config.dataset.iHiTOP.val_percentage + config.dataset.iHiTOP.test_percentage == 1.0
    total = len(seg_indices)

    train_size = int(config.dataset.iHiTOP.train_percentage * total)
    val_size = int(config.dataset.iHiTOP.val_percentage * total)
    test_size = total - train_size - val_size

    # this is the split used in the paper, randomly selected
    random_idxs = seg_indices
    if os.path.exists(config.dataset.iHiTOP.final_idxs):
        random_idxs = np.load(config.dataset.iHiTOP.final_idxs)
    else:
        random_idxs = seg_indices
        np.random.shuffle(random_idxs)
        print("Shuffled indices undefined! Defining now....")
        np.save(config.dataset.iHiTOP.final_idxs, random_idxs)
        print("Shuffled indices defined!")

    train_idxs = random_idxs[:train_size]
    val_idxs = random_idxs[train_size:train_size + val_size]
    test_idxs = random_idxs[train_size + val_size:]

    return iHiTOPDataset(hdf5_files, train_idxs, config, test=True), iHiTOPDataset(hdf5_files, val_idxs, config, test=True), iHiTOPDataset(hdf5_files, test_idxs, config, test=True), effective_seg_count #, train_list, val_list, test_list
