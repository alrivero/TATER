import h5py
import numpy as np
import cv2
import argparse
import os
import torch
import math
from pydub import AudioSegment
from transformers import Wav2Vec2Processor

def create_phoneme_group_timestamps(video_dict, phoneme_map):
    """
    Creates group timestamps given audio, video frames, dropped frames, and phoneme group mappings.
    Output: (GROUP ID, phoneme, start_idx, end_idx) where start_idx and end_idx are indices in img_tensor.
    """
    img_tensor = video_dict["img"]
    dropped_frames = video_dict.get("dropped_frames", torch.tensor([], dtype=torch.int32)).to(img_tensor.device)
    sequence_tensor = video_dict["audio_phonemes"][0]
    text_tensor = video_dict["text_phonemes"]
    audio_sample_rate = video_dict["audio_sample_rate"]  # Accessing attributes directly
    video_frame_rate = video_dict["fps"]  # Accessing attributes directly

    # Step 1: Measure the length of the audio in seconds
    audio_length_seconds = img_tensor.size(0) / video_frame_rate

    # Step 2: Calculate the number of video frames based on img_tensor shape and dropped frames
    total_frames = img_tensor.shape[0] + dropped_frames.shape[0]

    # Step 3: Calculate the timestamp for each video frame (including dropped frames)
    frame_timestamps = torch.linspace(0, total_frames / video_frame_rate, total_frames, device=img_tensor.device)

    # Step 5: Calculate phoneme durations in seconds
    phoneme_duration = audio_length_seconds / len(sequence_tensor)

    # Step 6: Create the array of tuples (group, phoneme, start_time, end_time) for valid groups (1-9)
    phoneme_id_to_group = phoneme_map
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

    # print(f"Original Indices: {video_group_indices}")

    # # Step 10: Finally, filter out false-positives using the text-based phonemes
    # text_phoneme_groups = [phoneme_id_to_group.get(int(x.item()), -1) for x in text_tensor]
    # final_group_indices = []
    # for group in text_phoneme_groups:
    #     for i, group_idx in enumerate(video_group_indices):
    #         if group_idx[0] == group:
    #             final_group_indices.append(group_idx)
    #             video_group_indices = video_group_indices[min(i + 1, len(video_group_indices)):]
    #             break
    
    # print(f"Final Indices: {final_group_indices}")
    return video_group_indices

# Define phoneme group colors (RGB format)
PHONEME_COLORS = {
    1: (255, 0, 0),   # Red
    2: (0, 255, 0),   # Green
    3: (0, 0, 255),   # Blue
    4: (255, 255, 0), # Yellow
    5: (255, 0, 255), # Magenta
    6: (0, 255, 255), # Cyan
    7: (128, 0, 128), # Purple
    8: (255, 165, 0), # Orange
    9: (0, 128, 128), # Teal
    10: (128, 128, 0) # Olive (for space/pause)
}

def create_phoneme_map():
    """
    Creates a mapping of phoneme IDs to phoneme groups.
    Uses Wav2Vec2Processor to get phoneme tokens and maps them to articulation-based groups.
    """
    phoneme_groups = {
        1: ["m", "b", "p"],
        2: ["f", "v"],
        3: ["th", "dh"],
        4: ["t", "d", "s", "z", "n", "l", "dx", "r"],
        5: ["sh", "zh"],
        6: ["j", "ch", "jh", "y"],
        7: ["k", "g", "ng", "w"],
        8: ["hh", "h#"],
        9: ["aa", "ae", "ah", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"],
        10: ["|"],
        11: ["spn"],
        12: ["[UNK]"],
        13: ["[PAD]"],
        14: ["<s>"],
        15: ["</s>"],
    }

    def get_phoneme_group_number(phoneme):
        """Determine the viseme group for a given phoneme."""
        for group_number, phonemes in phoneme_groups.items():
            if phoneme in phonemes:
                return group_number
        return -1

    processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    vocab_size = processor.tokenizer.vocab_size
    id_to_group = {}

    for idx in range(vocab_size):
        phoneme = processor.tokenizer.convert_ids_to_tokens([idx])[0]
        phoneme_group_number = get_phoneme_group_number(phoneme)
        id_to_group[idx] = phoneme_group_number

    return id_to_group

def tint_frame(frame, phoneme_group):
    """Applies a color tint to a frame based on the phoneme group."""
    if phoneme_group not in PHONEME_COLORS:
        print(phoneme_group, "WIIIE")
        return frame  # No tint if phoneme group is unknown
    
    color = np.array(PHONEME_COLORS[phoneme_group], dtype=np.uint8)
    tint_layer = np.full_like(frame, color, dtype=np.uint8)

    print("okqiojefoiwiefjOIEFJOIJEEIOWJ")
    
    return cv2.addWeighted(frame, 0.7, tint_layer, 0.3, 0)  # Blend 70% frame, 30% tint

def process_hdf5_video(hdf5_path, group_name, output_video, create_phoneme_group_timestamps):
    """Extracts a video and applies phoneme-based tinting using phoneme timestamps."""
    with h5py.File(hdf5_path, "r") as f:
        if group_name not in f:
            raise ValueError(f"Group '{group_name}' not found in the HDF5 file.")

        group = f[group_name]
        phoneme_map = create_phoneme_map()  # Generate phoneme-to-group mapping

        # Construct video_dict
        video_dict = {
            "img": torch.tensor(group["img"][:]),  
            "fps": group.attrs["fps"],
            "audio_sample_rate": group.attrs["sample_rate"],
            "audio_phonemes": torch.tensor(group["audio_phonemes"][:]),  
            "text_phonemes": torch.tensor(group["text_phonemes"][:]),  
            "dropped_frames": torch.tensor(group.get("removed_frames", []))
        }

        # Call the given function without modifications
        phoneme_timestamps = create_phoneme_group_timestamps(video_dict, phoneme_map)

        frames = video_dict["img"].numpy()
        fps = video_dict["fps"]
        sample_rate = video_dict["audio_sample_rate"]
        removed_frames = video_dict["dropped_frames"].numpy()

        # Convert float32 audio to int16 if needed
        audio = group["audio"][:]
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        frame_sample_length = int(sample_rate / fps)

        # Remove corresponding audio segments
        kept_audio = []
        last_sample = 0

        for idx in removed_frames:
            start_sample = idx * frame_sample_length
            if last_sample < start_sample:
                kept_audio.append(audio[last_sample:start_sample])
            last_sample = start_sample + frame_sample_length

        if last_sample < len(audio):
            kept_audio.append(audio[last_sample:])

        final_audio = np.concatenate(kept_audio) if kept_audio else np.array([], dtype=np.int16)

        # Convert to pydub format
        audio_segment = AudioSegment(
            final_audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

        audio_path = output_video.replace(".mp4", ".wav")
        audio_segment.export(audio_path, format="wav")

        # Define video properties
        height, width, _ = frames.shape[1:]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_video = output_video.replace(".mp4", "_temp.mp4")
        video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        # Apply phoneme-based tinting efficiently
        for group, phoneme, start_time, end_time in phoneme_timestamps:
            start_idx = max(0, int(start_time * fps))  # Convert time to frame index
            end_idx = min(len(frames), int(end_time * fps))  # Ensure within bounds

            for i in range(start_idx, end_idx):  # Directly iterate over affected frames
                frames[i] = tint_frame(frames[i], group)

        # Write all frames to video
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()

        # Merge video and audio using ffmpeg
        final_output = output_video.replace(".mp4", "_final.mp4")
        os.system(f'ffmpeg -y -r {fps} -i "{temp_video}" -i "{audio_path}" -c:v libx264 -c:a aac -strict experimental -shortest "{final_output}"')

        os.remove(temp_video)
        os.remove(audio_path)

        print(f"Video saved as: {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", type=str)
    parser.add_argument("group_name", type=str)
    parser.add_argument("output_video", type=str)

    args = parser.parse_args()
    process_hdf5_video(args.hdf5_path, args.group_name, args.output_video, create_phoneme_group_timestamps)