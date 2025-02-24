import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import debug
import traceback
import time
import warnings
from src.tater_encoder import TATEREncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from datasets.EXT_dataset import get_datasets_EXT
from datasets.mixed_dataset_sampler import MixedDatasetBatchSampler
import pickle
from pytorch3d.loss import chamfer_distance
import torchaudio
import torchaudio.transforms as T
from scipy.signal import resample
from scipy.interpolate import interp1d
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import math
import torch.nn.functional as F
import cv2
import ffmpeg
import soundfile as sf

SEG_LEN = 60
OVERLAP = 5

def parse_args():
    conf = OmegaConf.load(sys.argv[1])

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

def force_model_to_device(model, device):
    """
    Moves all parameters and buffers of the model to a specific device.

    Args:
        model (torch.nn.Module): The model to move.
        device (str): The target device (e.g., 'cuda:0').
    """
    device = torch.device(device)  # Ensure it's a proper device object

    # Move each parameter and buffer to the desired device
    for param in model.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)

    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)

    model.to(device)  # Ensure model is also set to the device
    print(f"✅ Model moved to {device}")

def load_EXT_dataloaders(config):
    # ----------------------- initialize datasets ----------------------- #
    train_dataset_EXT, val_dataset_EXT, test_dataset_EXT = get_datasets_EXT(config)
    dataset_percentages = {
        'EXT': 1.0
    }
    
    train_dataset = train_dataset_EXT
    sampler = MixedDatasetBatchSampler([
                                        len(train_dataset_EXT)
                                        ], 
                                       list(dataset_percentages.values()), 
                                       config.train.batch_size, len(train_dataset_EXT))
    def collate_fn(batch):
        combined_batch = {}
        for key in batch[0].keys():
            combined_batch[key] = [b[key] for b in batch]

        return combined_batch
    
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_EXT])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=config.train.batch_size, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_dataset_EXT


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
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
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

def save_video_with_overlay(
    all_imgs, all_renders, audio_tensor, sample_rate, output_path, fps=30, opacity=0.5, overlap=5
):
    """
    Saves an MP4 where the rendered images are overlaid on the original images with adjustable opacity.
    Also writes a stereo (N,2) audio tensor to the output video.

    Args:
        all_imgs (list of torch.Tensor): List of batches of original images (B, 3, N, M).
        all_renders (list of torch.Tensor): List of batches of rendered images (B, 3, N, M).
        audio_tensor (torch.Tensor): Stereo audio tensor (N, 2).
        sample_rate (int): Sample rate of the audio.
        output_path (str): Path to save the video file.
        fps (int): Frames per second.
        opacity (float): Opacity of the overlayed render (0 = no render, 1 = full render).
        overlap (int): Number of frames to ignore at the start of each batch except the first.
    """

    def to_uint8(tensor):
        """Converts PyTorch tensor from [0,1] or [-1,1] range to uint8 [0,255] for OpenCV."""
        tensor = tensor.clone().detach()

        if tensor.min() < 0:  # Convert [-1,1] range to [0,1]
            tensor = (tensor + 1) / 2  

        tensor = torch.clamp(tensor, 0, 1)
        tensor = (tensor * 255).byte()

        return tensor.permute(1, 2, 0).cpu().numpy()

    frames = []
    for batch_idx, (img_batch, render_batch) in enumerate(zip(all_imgs, all_renders)):
        batch_size = img_batch.shape[0]

        for b in range(batch_size):
            if batch_idx > 0 and b < overlap:
                continue

            img_np = to_uint8(img_batch[b])
            render_np = to_uint8(render_batch[b])

            if img_np.shape != render_np.shape:
                render_np = cv2.resize(render_np, (img_np.shape[1], img_np.shape[0]))

            overlayed = cv2.addWeighted(img_np, 1 - opacity, render_np, opacity, 0)

            frames.append(cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

    if len(frames) == 0:
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
    audio_np = audio_tensor

    # Convert int16 to float32 if necessary
    if np.issubdtype(audio_np.dtype, np.integer):
        audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max

    # Ensure audio is within [-1,1]
    audio_np = np.clip(audio_np, -1.0, 1.0)

    # Save audio as WAV
    sf.write(audio_file, audio_np, int(sample_rate))
    print(f"Audio saved at {audio_file}")

    # --- Corrected FFmpeg Call to Merge Video & Audio ---
    final_output = output_path

    ffmpeg_cmd = (
        ffmpeg
        .concat(
            ffmpeg.input(video_file).video,  # Ensure video stream
            ffmpeg.input(audio_file).audio,  # Ensure audio stream
            v=1, a=1  # Keep one video, one audio stream
        )
        .output(final_output, vcodec="libx264", acodec="aac", audio_bitrate="192k")
        .run(overwrite_output=True, quiet=True)
    )

    print(f"Final video with audio saved at {final_output}")

    # Cleanup temp files
    os.remove(video_file)
    os.remove(audio_file)

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
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    warnings.filterwarnings("ignore", message="GaussNoise could work incorrectly in ReplayMode for other input data")
    _, val_loader,  _ = load_EXT_dataloaders(config)

    def strip_exact_prefix(state_dict, prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    
    tater = TATEREncoder(config, n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    loaded_state_dict = torch.load(config.resume, map_location=config.device)
    filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k.startswith('tater')}
    filtered_state_dict = strip_exact_prefix(filtered_state_dict, "tater.")
    tater.load_state_dict(filtered_state_dict)
    tater.device = config.device
    force_model_to_device(tater, tater.device)
    tater.eval()

    flame = FLAME(n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    flame = flame.to(config.device)

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

    # Renderer
    renderer = Renderer(render_full_head=False)
    force_model_to_device(renderer, device)
    renderer.eval()

    # Assuming batchsize of 1
    for segment_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        if len(batch["img"][0]) == 0:
            print("Empty. Continue...")
            continue

        subbatch_start = 0
        audio_start = 0
        fps = batch["fps"][0].item()
        sample_rate = batch["audio_sample_rate"][0].item()

        all_imgs = []
        all_renders = []

        while subbatch_start <= len(batch["img"][0]):
            imgs = batch["img"][0][subbatch_start:subbatch_start + SEG_LEN].to(device)

            frames_sampled = len(imgs)
            duration_in_sec = round(frames_sampled / fps, 4)
            audio_sampled = int(sample_rate * duration_in_sec)

            audio_embed = get_audio_embed(
                batch["audio"][0][audio_start: audio_start + audio_sampled].float().mean(dim=-1).int().numpy(),
                audio_model,
                extracted_embeddings,
                frames_sampled,
                duration_in_sec,
                sample_rate,
                tater.device
            )

            phoneme_embed = get_phoneme_embed(
                batch["audio"][0][audio_start: audio_start + audio_sampled].float().mean(dim=-1).int(),
                phoneme_model,
                phoneme_processor,
                phoneme_map,
                frames_sampled,
                duration_in_sec,
                sample_rate,
                fps,
                tater.device
            )

            with torch.no_grad():
                series_len = [frames_sampled]
                all_params = tater([imgs], series_len, audio_batch=[audio_embed], phoneme_batch=phoneme_embed)

                flame_output = flame.forward(all_params)
                renderer_output = renderer.forward(flame_output['vertices'], all_params['cam'],
                                                        landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
                rendered_img = renderer_output['rendered_img']
                flame_output.update(renderer_output)

            all_imgs.append(imgs.cpu())
            all_renders.append(rendered_img.cpu())

            subbatch_start += SEG_LEN - OVERLAP
            audio_start += int(sample_rate * (SEG_LEN - OVERLAP) / fps)
        
        save_video_with_overlay(
            all_imgs,
            all_renders,
            batch["audio"][0].numpy(),
            sample_rate,
            f"{segment_idx}.mp4",
            fps,
            0.5,
            OVERLAP
        )

            

                

