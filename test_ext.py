import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import os
import torchvision.transforms as T
from src.tater_encoder import TATEREncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import torchaudio
import numpy as np
import math
import torch.nn.functional as F
import mediapipe as mp
import cv2
import ffmpeg
import soundfile as sf
import torchaudio.transforms as A
import torchaudio
import torchaudio.transforms as A
from scipy.signal import resample
from scipy.interpolate import interp1d
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import math
from datasets.EXT_dataset import get_datasets_EXT

SEG_LEN = 60
OVERLAP = 5
FACE_CROP = True


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove config file from args
    conf.merge_with_cli()
    return conf

def force_model_to_device(model, device):
    """
    Moves all parameters and buffers of the model to a specific device.
    """
    device = torch.device(device)

    for param in model.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)

    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)

    model.to(device)
    print(f"✅ Model moved to {device}")

def detect_face_and_crop(image_batch, detector, target_size=224, padding_value=0, scale_factor=1.7):
    """
    Detects faces using MediaPipe Face Detection and applies **tight, square cropping centered on the face**.
    If the crop is **out of bounds**, it is **padded** to maintain a centered, square image.

    **Scaling is performed by first centering the bounding box at the origin, scaling, and then shifting it back.**

    Args:
        image_batch (torch.Tensor): Batch of images **(B, 3, H, W)** or a single image **(3, H, W)** normalized to [0,1].
        detector: MediaPipe FaceDetection model (must be instantiated **outside** the function).
        target_size (int): Output size of the cropped face.
        padding_value (float): Pixel fill value for out-of-bounds regions (default: `0` for black padding).
        scale_factor (float): Multiplier for expanding/shrinking the detected face crop.

    Returns:
        cropped_images (torch.Tensor): Cropped faces resized to (B, 3, target_size, target_size) or (3, target_size, target_size) if input was a single image.
        bboxes (list of tuples): [(x_min, y_min, x_max, y_max)] for each image (None if no face detected).
        centers (list of tuples): [(x_center, y_center)] for each image (None if no face detected).
    """
    if image_batch.dim() == 3:  # Convert single image to batch (1, 3, H, W)
        image_batch = image_batch.unsqueeze(0)

    B, _, H, W = image_batch.shape
    cropped_images, bboxes, centers, paddings = [], [], [], []

    for i in range(B):
        image_np = (image_batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Convert to NumPy

        results = detector.process(image_np)  # Call process directly

        if results.detections:
            # **Get first detected face bounding box**
            bbox = results.detections[0].location_data.relative_bounding_box
            cx, cy = bbox.xmin + bbox.width / 2, bbox.ymin + bbox.height / 2

            x_min = bbox.xmin
            x_max = bbox.xmin + bbox.width
            y_min = bbox.ymin
            y_max = bbox.ymin + bbox.height

            # **Move bounding box center to (0,0) before scaling**
            x_min -= cx
            x_max -= cx
            y_min -= cy
            y_max -= cy

            # **Scale while keeping center fixed**
            x_min *= scale_factor
            x_max *= scale_factor
            y_min *= scale_factor
            y_max *= scale_factor

            # **Move back to original center**
            x_min += cx
            x_max += cx
            y_min += cy
            y_max += cy

            x_min *= W
            x_max *= W
            y_min *= H
            y_max *= H

            # **Convert to absolute pixel values**
            cx, cy = cx * W, cy * H

            # **Ensure integer indices for cropping**
            x_min, x_max = int(round(x_min)), int(round(x_max))
            y_min, y_max = int(round(y_min)), int(round(y_max))

            # **Step 1: Expand the bounding box to be square**
            box_width = x_max - x_min
            box_height = y_max - y_min
            max_side = max(box_width, box_height)  # Make the bounding box square

            # **Re-center the bounding box while making it square**
            x_min = int(round(cx - max_side / 2))
            x_max = int(round(cx + max_side / 2))
            y_min = int(round(cy - max_side / 2))
            y_max = int(round(cy + max_side / 2))

            # **Step 2: Clamp the modified bounding box to image bounds**
            x_min_clamped, x_max_clamped = max(0, x_min), min(W, x_max)
            y_min_clamped, y_max_clamped = max(0, y_min), min(H, y_max)

            # **Extract the portion inside the image bounds**
            crop = image_batch[i][:, y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped]

            # **Step 3: Compute padding needed for out-of-bounds areas**
            pad_top = abs(y_min - y_min_clamped) if y_min < 0 else 0
            pad_bottom = abs(y_max - y_max_clamped) if y_max > H else 0
            pad_left = abs(x_min - x_min_clamped) if x_min < 0 else 0
            pad_right = abs(x_max - x_max_clamped) if x_max > W else 0

            # **Step 4: Apply padding where necessary**
            crop_padded = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), value=padding_value)
            _, _, padded_W = crop_padded.shape
            crop_padded = T.functional.resize(crop_padded, (target_size, target_size))

            # **Step 5: Record scaled padding**
            crop_scale = target_size / padded_W

            # **Record results**
            cropped_images.append(crop_padded.cpu())
            bboxes.append((x_min_clamped, y_min_clamped, x_max_clamped, y_max_clamped))  # Bounding box before clamping
            centers.append((int(cx), int(cy)))  # Face center
            paddings.append((int(crop_scale * pad_top), int(crop_scale * pad_bottom), int(crop_scale * pad_left), int(crop_scale * pad_right)))
        else:
            # No face detected, return a blank image instead of an empty batch
            blank_image = torch.full((3, target_size, target_size), padding_value, dtype=torch.float32)
            cropped_images.append(blank_image)  # Keep batch shape consistent
            bboxes.append(None)
            centers.append(None)
            paddings.append(None)

    if not cropped_images:
        raise ValueError("No valid images found in batch!")

    cropped_images = torch.stack(cropped_images)  # Ensure all images have the same shape

    # If the input was a single image, remove batch dimension
    if B == 1:
        return cropped_images.squeeze(0), bboxes[0], centers[0], paddings[0]

    return cropped_images, bboxes, centers, paddings


def superimpose_image(original_image, overlay_image, bboxes, paddings, opacity=0.5, red_tint=0.2):
    """
    Superimposes rendered faces onto the original face bounding box.
    If no face was detected, it applies a red tint to the whole image.

    **Uses recorded padding values to crop the overlay before superimposing.**

    Args:
        original_image (torch.Tensor): **(B, 3, H, W)**
        overlay_image (torch.Tensor): **(B, 3, target_size, target_size)**
        bboxes (list of tuples): **Bounding boxes (x_min, y_min, x_max, y_max)**
        paddings (list of tuples): **Padding values from cropping (top, bottom, left, right)**
        opacity (float): **Blend factor for overlay**
        red_tint (float): **Amount of red tint to apply**

    Returns:
        modified_image (torch.Tensor): **Image with blended overlay or red tint if no face detected.**
    """
    B, _, H, W = original_image.shape
    modified_image = original_image.clone()

    for i in range(B):
        if bboxes[i] is None:
            modified_image[i][0] = torch.clamp(modified_image[i][0] + red_tint, 0, 1)
            continue  # Skip overlaying

        x_min, y_min, x_max, y_max = bboxes[i]
        pad_top, pad_bottom, pad_left, pad_right = paddings[i]

        # **Crop the overlay to remove padding before resizing**
        overlay_cropped = overlay_image[i][:, pad_top:-pad_bottom or None, pad_left:-pad_right or None]

        # **Resize cropped overlay to fit bounding box**
        overlay_resized = F.interpolate(overlay_cropped.unsqueeze(0), size=(y_max - y_min, x_max - x_min), mode="bilinear", align_corners=False).squeeze(0)

        modified_image[i, :, y_min:y_max, x_min:x_max] = opacity * overlay_resized + (1 - opacity) * original_image[i, :, y_min:y_max, x_min:x_max]

    return modified_image

def save_video(all_imgs, audio_tensor, sample_rate, output_path, fps=30):
    """
    Saves an MP4 video from `all_imgs` (B, 3, H, W) and synchronizes `audio_tensor` (N, 2).

    Args:
        all_imgs (torch.Tensor): Tensor of images, shape **(B, 3, H, W)**.
        audio_tensor (torch.Tensor): Dual-channel stereo audio tensor (N, 2).
        sample_rate (int): Audio sample rate.
        output_path (str): Path to save the video.
        fps (int): Frames per second.
    """

    def to_uint8(tensor):
        """Converts PyTorch tensor from [0,1] or [-1,1] to uint8 [0,255] for OpenCV."""
        tensor = tensor.clone().detach()

        if tensor.min() < 0:  # Convert [-1,1] range to [0,1]
            tensor = (tensor + 1) / 2  

        tensor = torch.clamp(tensor, 0, 1)
        tensor = (tensor * 255).byte()

        return tensor.permute(1, 2, 0).cpu().numpy()  # Convert (3, H, W) → (H, W, 3)

    # Convert images to OpenCV format
    frames = []
    B, _, H, W = all_imgs.shape

    for i in range(B):
        img_np = to_uint8(all_imgs[i])
        frames.append(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))  # Convert RGB → BGR for OpenCV

    if not frames:
        print("Error: No valid frames to write.")
        return

    # Define video writer
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_file = output_path.replace(".mp4", "_video.mp4")
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved at {video_file}")

    # --- Process Audio ---
    audio_file = output_path.replace(".mp4", ".wav")

    # Convert PyTorch tensor to NumPy
    audio_np = audio_tensor.cpu().numpy()

    # Convert int16 to float32 if necessary
    if np.issubdtype(audio_np.dtype, np.integer):
        audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max

    # Ensure audio is within [-1,1]
    audio_np = np.clip(audio_np, -1.0, 1.0)

    # Save audio as WAV
    sf.write(audio_file, audio_np, int(sample_rate))
    print(f"Audio saved at {audio_file}")

    # --- Merge Video & Audio with FFmpeg ---
    final_output = output_path

    ffmpeg_cmd = (
        ffmpeg
        .output(
            ffmpeg.input(video_file),
            ffmpeg.input(audio_file),
            final_output,
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="192k"
        )
    )

    ffmpeg_cmd.run(overwrite_output=True, quiet=True)
    print(f"Final video with audio saved at {final_output}")

    # Cleanup temp files
    os.remove(video_file)
    os.remove(audio_file)

class AlignmentError(Exception):
    """
    Custom exception raised when consonant alignment fails.
    """
    pass

def generate_phonemes(audio_segment, sample_rate, model, processor):
    """
    Aligns phonemes with audio segment using Wav2Vec2.

    Args:
        audio_segment (np.array): Audio segment as a numpy array.
        sample_rate (int): Sample rate of the audio.
        text (str): Text transcript.
        model: Pre-trained Wav2Vec2 model for phoneme recognition.
        processor: Wav2Vec2 processor.

    Returns:
        list: A list of tuples (start_time, end_time, phoneme, score).
    """
    try:
        
        # Prepare the audio for Wav2Vec2 input
        waveform = torch.tensor(audio_segment).unsqueeze(0).float().to(model.device)
        if sample_rate != processor.feature_extractor.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, processor.feature_extractor.sampling_rate)

        # Run through Wav2Vec2 model
        with torch.no_grad():
            logits = model(waveform).logits
        
        # Softmax to get probabilities
        audio_phonemes = torch.argmax(logits, axis=-1).cpu().numpy()

        return audio_phonemes

    except Exception as e:
        raise AlignmentError(f"Phoneme alignment failed: {e}")

def get_phoneme_embed(audio_data, phoneme_model, phoneme_processor, phoneme_map, total_frames, audio_length_seconds, audio_sample_rate, video_frame_rate, device):
    """
    Creates group timestamps given audio, video frames, dropped frames, and phoneme group mappings.
    Output: (GROUP ID, phoneme, start_idx, end_idx) where start_idx and end_idx are indices in img_tensor.
    """

    sequence_tensor = generate_phonemes(
        audio_data,
        audio_sample_rate,
        phoneme_model,
        phoneme_processor
    )
    sequence_tensor = sequence_tensor[0]

    dropped_frames = torch.tensor([], dtype=torch.int32)

    # Step 3: Calculate the timestamp for each video frame (including dropped frames)
    frame_timestamps = torch.linspace(0, total_frames / video_frame_rate, total_frames)

    # Step 5: Calculate phoneme durations in seconds
    phoneme_duration = audio_length_seconds / len(sequence_tensor)

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
        start_idx = max(0, min(start_idx, total_frames - 1))
        end_idx = max(0, min(end_idx, total_frames - 1))

        if start_idx == end_idx:
            continue
        video_group_indices.append((group, phoneme, start_idx, end_idx))
    
    # Step 10: Generate phoenme embedding using timestamps
    all_phoneme_onehot = torch.zeros(total_frames, 44).to(device)
    series_start = 0
    for _, phoneme_id, s_idx, e_idx in video_group_indices:
        # print(i, phoneme_id, s_idx, e_idx, all_phoneme_onehot.shape)
        phoneme_embed = F.one_hot(torch.tensor([phoneme_id] * (e_idx - s_idx)).to(device), num_classes=44)
        # print(phoneme_embed.shape, all_phoneme_onehot.shape, series_start + s_idx, series_start + e_idx)

        all_phoneme_onehot[series_start + s_idx:series_start + e_idx] += phoneme_embed
        # print("UWU", all_phoneme_onehot)
    
    return all_phoneme_onehot

def get_audio_embed(audio_data, audio_model, extracted_embeddings, num_frames, duration, sample_rate, device):
    # Normalize and resample audio
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    # Convert audio to tensor
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    if sample_rate != 16000:
        resampler = A.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = resampler(audio_tensor)

    audio_tensor = audio_tensor.to(device)

    MIN_LENGTH = 768
    if audio_tensor.shape[1] < MIN_LENGTH:
        padding_needed = MIN_LENGTH - audio_tensor.shape[1]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding_needed))

    # Forward pass through the model
    with torch.no_grad():
        #print("audio tensor shape:", audio_tensor.shape)
        wav2vec_output = audio_model(audio_tensor)
        #embeddings = wav2vec_output[0].cpu().numpy()
        embeddings = extracted_embeddings['final_layer_norm'].cpu().numpy()


    # Convert to frame-level embeddings
    embedding_dim = embeddings.shape[2]
    wav2vec_embeddings = embeddings.squeeze(0)

    segment_times = np.linspace(0, duration, num=wav2vec_embeddings.shape[0])
    frame_times = np.linspace(0, duration, num=num_frames)
    frame_level_embeddings = np.zeros((num_frames, embedding_dim), dtype="float32")

    for i in range(embedding_dim):
        if wav2vec_embeddings.shape[0] == 1:
            frame_level_embeddings[:, i] = wav2vec_embeddings[:, i]
        else:
            interp_func = interp1d(segment_times, wav2vec_embeddings[:, i], kind='linear', fill_value="extrapolate")
            frame_level_embeddings[:, i] = interp_func(frame_times)
    
    return torch.tensor(frame_level_embeddings).to(device)

def create_phoneme_map(phoneme_processor):
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

        vocab_size = phoneme_processor.tokenizer.vocab_size

        # Map each phoneme ID to its corresponding viseme group
        for idx in range(vocab_size):
            phoneme = phoneme_processor.tokenizer.convert_ids_to_tokens([idx])[0]
            phoneme_group_number = get_phoneme_group_number(phoneme)
            id_to_group[idx] = phoneme_group_number

        return id_to_group


if __name__ == '__main__':
    config = parse_args()

    train_loader, val_loader, _ = get_datasets_EXT(config)

    def strip_exact_prefix(state_dict, prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # Initialize models
    tater = TATEREncoder(config, n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    loaded_state_dict = torch.load(config.resume, map_location=config.device)
    filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k.startswith('tater')}
    filtered_state_dict = strip_exact_prefix(filtered_state_dict, "tater.")
    tater.load_state_dict(filtered_state_dict, strict=False)
    tater.device = config.device
    force_model_to_device(tater, tater.device)
    tater.eval()

    flame = FLAME(n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    flame = flame.to(config.device)
    force_model_to_device(flame, config.device)
    flame.eval()

    # Load in Wav2Vec audio model
    # Initialize Wav2Vec 2.0 model from torchaudio
    device = tater.device
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_100H
    audio_model = bundle.get_model().to(device)
    audio_model.eval()

    # Define a dictionary to store the output from the final_layer_norm
    extracted_embeddings = {}

    # Hook function to save output
    def hook_fn(module, input, output):
        extracted_embeddings['final_layer_norm'] = output

    final_layer = audio_model.encoder.transformer.layers[-1]  # Get the last encoder layer
    final_layer.final_layer_norm.register_forward_hook(hook_fn)

    # Also need the phoneme model
    model_id = "vitouphy/wav2vec2-xls-r-300m-phoneme"
    phoneme_processor = Wav2Vec2Processor.from_pretrained(model_id)
    phoneme_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(tater.device)
    phoneme_model.eval()

    phoneme_map = create_phoneme_map(phoneme_processor)

    renderer = Renderer(render_full_head=False)
    force_model_to_device(renderer, config.device)
    renderer.eval()

    # Face Detection
    if FACE_CROP:
        face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Process validation set
    for segment_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        if segment_idx >= 6:
            break
        if len(batch["img"][0]) == 0:
            continue

        subbatch_start = 0
        audio_start = 0
        fps = batch["fps"][0].item()
        sample_rate = batch["audio_sample_rate"][0].item()

        all_imgs, all_renders, all_crops, all_bbxs, all_paddings = [], [], [], [], []

        while subbatch_start <= len(batch["img"]):
            imgs = batch["img"][subbatch_start:subbatch_start + SEG_LEN].to(config.device)

            frames_sampled = len(imgs)
            duration_in_sec = round(frames_sampled / fps, 4)
            audio_sampled = int(sample_rate * duration_in_sec)

            audio_embed = get_audio_embed(
                batch["audio"][audio_start: audio_start + audio_sampled].float().mean(dim=-1).int().numpy(),
                audio_model,
                extracted_embeddings,
                frames_sampled,
                duration_in_sec,
                sample_rate,
                tater.device
            )

            if FACE_CROP:
                cropped_imgs, bboxes, _, paddings = detect_face_and_crop(imgs, face_detector)
                cropped_imgs = cropped_imgs.to(config.device) 
            else:
                cropped_imgs = imgs
                bboxes = [None] * len(imgs)
                paddings = [None] * len(imgs)

            with torch.no_grad():
                series_len = [frames_sampled]
                all_params = tater([cropped_imgs], series_len, audio_batch=[audio_embed])

                flame_output = flame.forward(all_params)
                renderer_output = renderer.forward(flame_output['vertices'], all_params['cam'],
                                                        landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
                rendered_imgs = renderer_output['rendered_img']
                flame_output.update(renderer_output)

            curr_overlap = OVERLAP if subbatch_start != 0 else 0

            if FACE_CROP:
                all_imgs.append(imgs[curr_overlap:].cpu())  # Keep original unmodified
                all_bbxs += bboxes[curr_overlap:]
                all_paddings += paddings[curr_overlap:]
            all_renders.append(rendered_imgs[curr_overlap:].cpu())
            all_crops.append(cropped_imgs[curr_overlap:].cpu())

            subbatch_start += SEG_LEN - OVERLAP
            audio_start += int(sample_rate * (SEG_LEN - OVERLAP) / fps)

        if FACE_CROP:
            final_imgs = superimpose_image(torch.cat(all_imgs), torch.cat(all_renders), all_bbxs, all_paddings, opacity=0.75, red_tint=0.0)
            final_imgs_crop = superimpose_image(torch.cat(all_imgs), torch.cat(all_crops), all_bbxs, all_paddings, opacity=0.5, red_tint=0.25)

        save_video(final_imgs, batch["audio"], sample_rate, f"{segment_idx}_CARA.mp4", fps=25)
        save_video(final_imgs_crop, batch["audio"], sample_rate, f"{segment_idx}_CARA_CROP.mp4", fps=25)