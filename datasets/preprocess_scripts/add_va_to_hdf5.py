import os
import re
import h5py
import numpy as np
import torch
import cv2
import csv
import argparse
import pickle
import json
import time
import librosa
import random
import torchaudio
import albumentations as A
import torchvision.transforms as transforms
import pronouncing
import pandas as pd
from torchvision.transforms.functional import resize
from skimage import transform as trans
from skimage.transform import estimate_transform, warp
from multiprocessing import Pool, cpu_count, Lock
from torch.multiprocessing import Process, Queue
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from multiprocessing import Manager
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Initialize a global tqdm lock
tqdm.set_lock(Lock())

# These are the indices of the mediapipe landmarks that correspond to the mediapipe landmark barycentric coordinates provided by FLAME2020
mediapipe_indices = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
        55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
       381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
       144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
       168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
         0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
        87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
       308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
       415]

arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return frame

def load_frames_from_images_folder(images_folder):
    frames = []
    for image_file in sorted(os.listdir(images_folder)):
        image_path = os.path.join(images_folder, image_file)
        frame = cv2.imread(image_path)
        frame = preprocess_frame(frame)
        frames.append(frame)
    frames = np.array(frames)
    tensor = torch.tensor(frames)
    return tensor

def create_mask(landmarks, shape):
    landmarks = landmarks.astype(np.int32)[...,:2]
    hull = cv2.convexHull(landmarks)
    mask = np.ones(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 0)

    return mask

def crop_face(frame, landmarks, scale=1.0):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, size - 1], [size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform, size

def crop_face_final(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if len(valid_frames_idx) > 1:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

def processs_frame(image, landmarks_fan, landmarks_mediapipe, crop_res):
    # Find the median resolution of all frames
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if landmarks_fan is None:
        flag_landmarks_fan = False
        landmarks_fan = np.zeros((68, 2))
    else:
        flag_landmarks_fan = True
        if len(landmarks_fan.shape) == 3:
            landmarks_fan = landmarks_fan[0]

    tform = crop_face_final(image, landmarks_mediapipe, 2.2, image_size=crop_res)

    landmarks_mediapipe = landmarks_mediapipe[..., :2]

    cropped_image = warp(image, tform.inverse, output_shape=(crop_res, crop_res), preserve_range=True).astype(np.uint8)
    cropped_landmarks_fan = np.dot(tform.params, np.hstack([landmarks_fan, np.ones([landmarks_fan.shape[0], 1])]).T).T
    cropped_landmarks_fan = cropped_landmarks_fan[:, :2]

    cropped_landmarks_mediapipe = np.dot(tform.params, np.hstack([landmarks_mediapipe, np.ones([landmarks_mediapipe.shape[0], 1])]).T).T
    cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[:, :2]

    # find convex hull for masking the face 
    hull_mask = create_mask(cropped_landmarks_mediapipe, (crop_res, crop_res))

    cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[mediapipe_indices, :2]

    # ----------- mica images ---------------- #
    landmarks_arcface_crop = landmarks_fan[[36, 45, 32, 48, 54]].copy()
    landmarks_arcface_crop[0] = (landmarks_fan[36] + landmarks_fan[39]) / 2
    landmarks_arcface_crop[1] = (landmarks_fan[42] + landmarks_fan[45]) / 2

    tform = estimate_norm(landmarks_arcface_crop, 112)

    image = image / 255.
    mica_image = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)
    mica_image = mica_image.transpose(2, 0, 1)

    image = cropped_image
    landmarks_fan = cropped_landmarks_fan
    landmarks_mediapipe = cropped_landmarks_mediapipe
    hull_mask = hull_mask
    mica_image = mica_image

    data_dict = {
        'img': image[None],
        'landmarks_fan': landmarks_fan[..., :2][None],
        'flag_landmarks_fan': flag_landmarks_fan,  # if landmarks are not available
        'landmarks_mp': landmarks_mediapipe[..., :2][None],
        'mask': hull_mask[None],
        'img_mica': mica_image[None]
    }

    return data_dict

def get_median_resolution(images):
    resolutions = [(img.shape[1], img.shape[2]) for img in images]  # Get (height, width) for each image
    heights, widths = zip(*resolutions)
    median_height = int(torch.median(torch.tensor(heights)).item())
    median_width = int(torch.median(torch.tensor(widths)).item())
    return max(median_height, median_width)

def resize_images_to_median(images, median_height, median_width):
    resized_images = []
    resize_transform = transforms.Resize((median_height, median_width))
    for img in images:
        resized_img = resize_transform(img)
        resized_images.append(resized_img)
    return resized_images

def get_crop_res(image, landmarks_fan, landmarks_mediapipe):
    # Find the median resolution of all frames
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if landmarks_fan is None:
        landmarks_fan = np.zeros((68, 2))
    else:
        if len(landmarks_fan.shape) == 3:
            landmarks_fan = landmarks_fan[0]

    _, new_size = crop_face(image, landmarks_mediapipe, 1.0)

    return new_size

def process_segment(frames, all_fan_landmarks, all_mp_landmarks):
    # Constructing this main dict of all data associated with the video segment
    data_dict = {
        'img': [],
        'landmarks_fan': [],
        'flag_landmarks_fan': [],  # if landmarks are not available
        'landmarks_mp': [],
        'mask': [],
        'img_mica': []
    }

    # Determine the appropriate crop resolution we should use for all frames in this segment
    crop_reses = []
    for img, fan_lmks, mp_lmks in zip(frames, all_fan_landmarks, all_mp_landmarks):
        crop_reses.append(get_crop_res(img, fan_lmks, mp_lmks))
    final_crop_res = int(np.median(np.array(crop_reses)))

    # Process all frames
    for frame, fan_lmks, mp_lmks in zip(frames, all_fan_landmarks, all_mp_landmarks):
        frame_data = processs_frame(frame, fan_lmks, mp_lmks, final_crop_res)

        for key in data_dict.keys():
            data_dict[key].append(frame_data[key])

    # Combine the frame data of all frames into the main dict
    for key in data_dict.keys():
        if key != 'flag_landmarks_fan':
            data_dict[key] = np.concatenate(data_dict[key])

    return data_dict

def define_segments_old(transcript_path):
    all_segments = []
    try:
        # Try loading as JSON first
        with open(transcript_path, 'r') as file:
            data = json.load(file)
            segments = data["segments"]
            if len(segments) == 0:
                return None

            percent_99 = np.percentile(np.array([x["no_speech_prob"] for x in segments]), 99)

            for segment in data["segments"]:
                if "?" in segment["text"] or segment["no_speech_prob"] >= percent_99:
                    continue
                all_segments.append(segment)

            return all_segments

    except json.JSONDecodeError:
        # If JSON fails, try loading as CSV
        try:
            with open(transcript_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row

                # Collect segments with speaker information
                all_segments = []
                for row in reader:
                    speaker, start_time, end_time, text = row

                    # Skip segments that contain a question mark
                    if "?" in text:
                        continue

                    start_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(':'))))
                    end_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(':'))))
                    all_segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                        "tokens": None,
                    })

                # Calculate pause lengths between segments
                pause_lengths = np.array([all_segments[i]["start"] - all_segments[i-1]["end"] for i in range(1, len(all_segments))])

                # Calculate median and MAD of pause lengths
                median_pause = np.median(pause_lengths)
                mad_pause = np.median(np.abs(pause_lengths - median_pause))

                # Assign weights to pauses based on their lengths
                pause_weights = []
                for i in range(1, len(all_segments)):
                    pause_length = all_segments[i]["start"] - all_segments[i-1]["end"]
                    if abs(pause_length - median_pause) <= 1.25 * mad_pause:
                        weight = 0.20  # Cap at 0.75 for pauses within 1.25 MAD
                    else:
                        # Logarithmic decay for other pauses, scaled between 0.05 and 0.75
                        distance_from_median = abs(pause_length - median_pause) / mad_pause  # Normalize distance
                        weight = 0.20 * np.exp(-np.log(distance_from_median))  # Logarithmic decay scaled to max 0.75
                        weight = max(weight, 0.01)  # Ensure a minimum weight of 0.01
                    pause_weights.append(weight)

                # Insert pseudo segments with probabilities based on weights
                new_segments = []
                silent_count = 0
                i = 0
                while i < len(all_segments):
                    all_segments[i]["id"] = i
                    new_segments.append(all_segments[i])
                    if i < len(all_segments) - 1 and np.random.rand() < pause_weights[i]:  # Sample based on weight
                        i += 1
                        new_segments.append({
                            "start": all_segments[i]["end"],
                            "end": all_segments[i + 1]["start"],
                            "text": "",
                            "tokens": np.array([-1]),
                            "id": i
                        })
                        silent_count += 1
                    i += 1
                print(f"Spoken Segments: {len(all_segments)}, Silent Segments: {silent_count}")

                return new_segments

        except Exception as e:
            print(f"Failed to load transcript file: {e}")
            return None

def define_segments(transcripts_df, va_df):
    new_segments = {}
    transcripts_df = transcripts_df[transcripts_df["speaker_role"] == "participant"]

    for _, row in transcripts_df.iterrows():  # Use iterrows to iterate over rows
        segment_dict = {}

        segment_dict["start"] = pd.to_timedelta(row["start"]).total_seconds()
        segment_dict["end"] = pd.to_timedelta(row["end"]).total_seconds()
        segment_dict["text"] = row["message"]
        segment_dict["id"] = row["index"]
        segment_dict["tokens"] = None

        va_rows = va_df[va_df["index"] == row["index"]]
        va_rows = va_rows.sort_values(by='feat').reset_index(drop=True)

        segment_dict["valence_arousal"] = torch.tensor(va_rows['value'].values, dtype=torch.float32)
        new_segments[row["index"]] = segment_dict
    
    return new_segments

class AlignmentError(Exception):
    """
    Custom exception raised when consonant alignment fails.
    """
    pass

class AlignmentError(Exception):
    """
    Custom exception raised when consonant alignment fails.
    """
    pass

# ARPAbet to Wav2Vec2-compatible phoneme conversion (finalized mapping)
def normalize_arpabet(phoneme):
    # Remove stress markers (e.g., AO1 -> ao)
    phoneme = re.sub(r'[0-9]', '', phoneme).lower()

    # Map ARPAbet phonemes to Wav2Vec2's phoneme set
    arpabet_to_wav2vec2 = {
        'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'aw': 'aw', 'ay': 'ay',
        'b': 'b', 'ch': 'ch', 'd': 'd', 'dh': 'dh', 'dx': 'd', 'eh': 'eh',
        'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g', 'hh': 'hh',
        'ih': 'ih', 'iy': 'iy', 'jh': 'jh', 'k': 'k', 'l': 'l',
        'm': 'm', 'n': 'n', 'ng': 'ng', 'ow': 'ow', 'oy': 'oy',
        'p': 'p', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't',
        'th': 'th', 'uh': 'uh', 'uw': 'uw', 'v': 'v', 'w': 'w',
        'y': 'y', 'z': 'z', 'spn': 'spn', 'h#': 'hh',  # h# mapped to hh (glottal stop)
        '|': '|', '[unk]': '[UNK]', '[pad]': '[PAD]'  # Handle special tokens if needed
    }

    return arpabet_to_wav2vec2.get(phoneme, '[UNK]')

def build_phoneme_to_id_dict(processor):
    """
    Build a dictionary mapping each phoneme to its corresponding token ID.
    """
    phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(processor.tokenizer.convert_ids_to_tokens(range(processor.tokenizer.vocab_size)))}
    return phoneme_to_id

def text_to_phoneme_ids(text, phoneme_to_id):
    """
    Convert text to a list of phoneme IDs using ARPAbet-to-IPA mapping
    and the phoneme-to-ID dictionary. Removes punctuation before processing.
    """
    # Remove punctuation from the text
    text = re.sub(r'[^\w\s]', '', text)
    
    words = text.split()
    phoneme_ids = []
    for word in words:
        phoneme_list = pronouncing.phones_for_word(word.lower())
        if phoneme_list:
            normalized_phonemes = [normalize_arpabet(p) for p in phoneme_list[0].split()]
            ids = [phoneme_to_id.get(phoneme, phoneme_to_id['[UNK]']) for phoneme in normalized_phonemes]
            phoneme_ids.extend(ids)
        phoneme_ids.append(phoneme_to_id['|'])  # Add a word boundary as '|'
    return phoneme_ids[:-1]  # Remove the trailing boundary ID


def generate_phonemes(audio_segment, sample_rate, text, model, processor):
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
        # Convert the text to phonemes (e.g., using an external method)
        phoneme_to_id = build_phoneme_to_id_dict(processor)
        text_phonemes = np.array(text_to_phoneme_ids(text, phoneme_to_id))
        
        # Prepare the audio for Wav2Vec2 input
        waveform = torch.tensor(audio_segment).unsqueeze(0).float().to(model.device)
        if sample_rate != processor.feature_extractor.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, processor.feature_extractor.sampling_rate)

        # Run through Wav2Vec2 model
        with torch.no_grad():
            logits = model(waveform).logits
        
        # Softmax to get probabilities
        audio_phonemes = torch.argmax(logits, axis=-1).cpu().numpy()

        return text_phonemes, audio_phonemes

    except Exception as e:
        raise AlignmentError(f"Phoneme alignment failed: {e}")


def extract_landmarks(all_frames, fan_face_detector, fan_landmark_detector, mp_landmark_detector):
    # Extract landmarks from this video segment
    all_mp_landmarks = []
    all_fan_landmarks = []
    bad_frames = []
    for i, frame in enumerate(all_frames):
        # Fan landmarks
        fan_detected_faces = fan_face_detector(frame, rgb=False)
        fan_landmarks, _ = fan_landmark_detector(frame, fan_detected_faces, rgb=False)

        if len(fan_landmarks) == 0:
            bad_frames.append(i)
            continue

        # Mediapipe landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = mp_landmark_detector.detect(mp_image)

        if not detection_result.face_landmarks:
            bad_frames.append(i)
            continue

        mp_landmarks = detection_result.face_landmarks[0]
        mp_landmarks = np.array([[landmark.x * mp_image.width, landmark.y * mp_image.height, landmark.z] for landmark in mp_landmarks])

        all_fan_landmarks.append(fan_landmarks)
        all_mp_landmarks.append(mp_landmarks)

    return all_fan_landmarks, all_mp_landmarks, bad_frames

def get_num_frames(start, end, cap):
    return min(end, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) - start

def extract_segment_data_split_video(segment, root_directory, video_paths, fan_face_detector, fan_landmark_detector, mp_landmark_detector, split_min_len=20):
    # Seek video based on segment duration and gather frames
    cap = cv2.VideoCapture(os.path.join(root_directory, "videos", video_paths[-1]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start = int(fps * segment["start"])
    end = int(fps * segment["end"])
    cap.release()
    video_len = int(split_min_len * 60 * fps)

    all_frames = []
    if int(start // video_len) == int(end // video_len):
        cap = cv2.VideoCapture(os.path.join(root_directory, "videos", video_paths[int(start // video_len)]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start % video_len)
        for frame_num in range(start % video_len, end % video_len):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise Exception("Error while reading video frames!")
            all_frames.append(frame)
        cap.release()
    else:
        # Process first part
        cap1 = cv2.VideoCapture(os.path.join(root_directory, "videos", video_paths[int(start // video_len)]))
        cap1.set(cv2.CAP_PROP_POS_FRAMES, start % video_len)
        cap1_len = video_len - (start % video_len)  # Number of frames to read in the first part
        for _ in range(cap1_len):
            ret, frame = cap1.read()
            if not ret:
                cap1.release()
                raise Exception("Error while reading video frames (1)!")
            all_frames.append(frame)
        cap1.release()

        # Process second part
        cap2 = cv2.VideoCapture(os.path.join(root_directory, "videos", video_paths[int((start // video_len) + 1)]))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2_len = int(end % video_len)  # Number of frames to read in the second part
        for _ in range(cap2_len):
            ret, frame = cap2.read()
            if not ret:
                cap2.release()
                raise Exception("Error while reading video frames (2)!")
            all_frames.append(frame)
        cap2.release()

    # Extract landmarks and handle bad frames
    all_fan_landmarks, all_mp_landmarks, bad_frames = extract_landmarks(
        all_frames,
        fan_face_detector,
        fan_landmark_detector,
        mp_landmark_detector
    )

    all_frames = [x for i, x in enumerate(all_frames) if i not in bad_frames]
    return all_frames, all_fan_landmarks, all_mp_landmarks, bad_frames, fps

def extract_segment_data_single_video(segment, root_directory, video_path, fan_face_detector, fan_landmark_detector, mp_landmark_detector):
    # Seek video based on segment duration and gather frames
    cap = cv2.VideoCapture(os.path.join(root_directory, "videos", video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start = int(fps * segment["start"])
    end = int(fps * segment["end"])

    all_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame_num in range(start, end):
        ret, frame = cap.read()

        if not ret:
            cap.release()
            raise Exception("Error while reading video frames!")

        all_frames.append(frame)
    cap.release()

    all_fan_landmarks, all_mp_landmarks, bad_frames = extract_landmarks(
        all_frames,
        fan_face_detector,
        fan_landmark_detector,
        mp_landmark_detector
    )

    all_frames = [x for i, x in enumerate(all_frames) if i not in bad_frames]
    return all_frames, all_fan_landmarks, all_mp_landmarks, bad_frames, fps

def extract_audio_segment(segment, audio, sample_rate, bad_frames, vid_fps):
    # Get our audio segment...
    start = int(sample_rate * segment["start"])
    end = int(sample_rate * segment["end"])
    audio_segment = audio[start:end]

    # ...but remove any parts that have no associated frame
    bad_samples = []
    bad_sub_segments = []
    for bad_frame in bad_frames:
        start_sample = min(len(audio_segment), int(bad_frame / vid_fps * sample_rate))
        end_sample = min(len(audio_segment), int((bad_frame / vid_fps + 1 / vid_fps) * sample_rate))

        if not bad_samples:
            bad_samples.extend([start_sample, end_sample])
        elif bad_samples[-1] >= start_sample:
            bad_samples[-1] = end_sample
        else:
            bad_sub_segments.append(tuple(bad_samples))
            bad_samples = [start_sample, end_sample]

    if bad_samples:
        bad_sub_segments.append(tuple(bad_samples))

    if bad_sub_segments:
        bad_sub_segments = [np.arange(s, e) for s, e in bad_sub_segments]
        bad_sub_segments = np.concatenate(bad_sub_segments)
        audio_segment = np.delete(audio_segment, bad_sub_segments)
    
    return audio_segment

def log_completion(log_file_path, video_title, duration, lock):
    with lock:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Completed {video_title} in {duration:.2f} seconds\n")

def strings_to_bytes(strings):
    # # Convert Unicode strings to ASCII (or another encoding)
    # ascii_strings = [s.encode('ascii', 'ignore').decode('ascii') for s in strings] 
    # # Now use your original code
    return np.array([s.encode('utf-8') for s in strings], dtype='S')  

def check_hdf5_integrity(file_path, sample_size=5):
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False
    
    try:
        with h5py.File(file_path, 'r') as f:
            datasets = []

            def collect_datasets(name, node):
                if isinstance(node, h5py.Dataset):
                    datasets.append(name)
            
            f.visititems(collect_datasets)
            
            if not datasets:
                print("No datasets found in the file.")
                return False
            
            # Sample a subset of datasets
            sample_size = min(sample_size, len(datasets))
            sample_datasets = random.sample(datasets, sample_size)
            
            for ds in sample_datasets:
                _ = f[ds][()]  # Read the dataset to ensure it can be accessed
            
        return True
    except Exception as e:
        print(f"File integrity check failed: {e}")
        return False
    
def extract_audio_from_video(video_path, segment_start, segment_end, bad_frames, vid_fps):
    """
    Extracts audio from a video file using moviepy for a given time segment, and retrieves the sample rate.
    
    Args:
        video_path (str): Path to the video file.
        segment_start (float): Start time of the audio segment in seconds.
        segment_end (float): End time of the audio segment in seconds.
        
    Returns:
        np.array: The audio segment as a numpy array.
        int: The sample rate of the extracted audio.
    """
    # Open the video file using moviepy
    with VideoFileClip(video_path) as video_clip:
        audio_clip = video_clip.audio.subclip(segment_start, segment_end)
        audio_segment = audio_clip.to_soundarray()
        audio_segment = np.mean(audio_segment, axis=1)  
        sample_rate = audio_clip.fps  # Get the actual sample rate from the audio clip
    
    return audio_segment, sample_rate

def process_video(root_directory, video_title, transcript_df, va_df, out_dir, log_file_path, gpu_ID, process_ID, lock, gpu_queue, include_audio=True):
    start_time = time.time()  # Start timing
    temp_hdf5_path = os.path.join(out_dir, f"{video_title}.h5")

    # Check if an hdf5 already exists and is healthy
    if not check_hdf5_integrity(temp_hdf5_path):
        print(f"No healthy HDF5 for {video_title} found. Skipping.")
        gpu_queue.put((gpu_ID, process_ID))
        return None

    # Open the HDF5 file in read-write mode
    with h5py.File(temp_hdf5_path, 'r+') as temp_hdf5_file:
        # Define all segments present in the video
        video_segments = define_segments(transcript_df, va_df)
        if video_segments is None:
            print(f"No segments for {video_title}!")
            gpu_queue.put((gpu_ID, process_ID))
            return None


        # Process each segment
        for group_id in temp_hdf5_file.keys():
            try:
                group_data = temp_hdf5_file[group_id]
                if "og_id" not in group_data.attrs.keys():
                    continue
                og_id = str(group_data.attrs["og_id"])  # Retrieve the original ID

                # Create or update 'valence_arousal' dataset
                if "valence_arousal" not in group_data and len(video_segments[og_id]["valence_arousal"]) != 0:
                    group_data.create_dataset("valence_arousal", data=video_segments[og_id]["valence_arousal"])
                elif len(video_segments[og_id]["valence_arousal"]) != 0:
                    group_data["valence_arousal"][()] = video_segments[og_id]["valence_arousal"]
                elif "valence_arousal" in group_data:
                    del temp_hdf5_file[f"{og_id}/valence_arousal"]

            except Exception as e:
                print(f"Problem processing segment {group_id} from {video_title}!")
                print(e)

    # Log the completion details
    temp_hdf5_file.close()
    duration = time.time() - start_time  # Calculate duration
    log_completion(log_file_path, video_title, duration, lock)
    gpu_queue.put((gpu_ID, process_ID))

def save_to_hdf5(root_directory, out_dir, transcript_file, va_file, log_file_path, gpus, tasks_per_gpu, include_audio):
    active_processes = []
    for _ in range(max(gpus) + 1):
        active_processes.append([])

    gpu_queue = Queue()
    for id, tasks in zip(gpus, tasks_per_gpu):
        for i in range(tasks):
            gpu_queue.put((id, i))
            active_processes[id].append(None)

    # Open our transcript file containing all transcripts
    # Open our transcript, va, roberta file containing all extra data
    all_transcripts_df = pd.read_csv(transcript_file)
    all_va_df = pd.read_csv(va_file)

    # Filter out UNK entries early
    all_transcripts_df = all_transcripts_df[all_transcripts_df['user_id'] != 'UNK'].copy()
    all_va_df = all_va_df[~all_va_df['group_id'].str.contains('UNK', na=False)].copy()

    # Strip the prefix from filenames (unless it starts with P)
    all_transcripts_df["filename_stripped"] = all_transcripts_df['filename_stripped'].apply(
        lambda x: "_".join(x.split("_")[1:]) if not x.startswith("P") else x
    )

    # Get unique filenames and PIDs
    transcript_unique_idxs = all_transcripts_df.drop_duplicates(subset=['filename_stripped'], keep='first').index
    transcript_titles = all_transcripts_df["filename_stripped"].iloc[transcript_unique_idxs]
    PIDs = all_transcripts_df["user_id"].iloc[transcript_unique_idxs]

    # Split message_id in transcripts only where it has 2 underscores (3 parts)
    transcript_mask = all_transcripts_df['message_id'].str.count('_') == 2
    all_transcripts_df = all_transcripts_df[transcript_mask].copy()

    split_transcripts = all_transcripts_df['message_id'].str.split('_', expand=True)
    split_transcripts.columns = ['PID', 'speaker_id', 'index']
    all_transcripts_df[['PID', 'speaker_id', 'index']] = split_transcripts

    # Same for group_id in all_va_df
    va_mask = all_va_df['group_id'].str.count('_') == 2
    all_va_df = all_va_df[va_mask].copy()

    split_va = all_va_df['group_id'].str.split('_', expand=True)
    split_va.columns = ['PID', 'speaker_id', 'index']
    all_va_df[['PID', 'speaker_id', 'index']] = split_va

    # Create a dictionary mapping base name (without .csv) to PID
    transcript_dict = {
        os.path.splitext(t)[0]: pid
        for t, pid in zip(transcript_titles, PIDs)
    }

    # Build list of (basename, PID) tuples where matched
    zipped_titles = [(base, transcript_dict[base]) for base in transcript_dict.keys()]

    manager = Manager()
    lock = manager.Lock()  # Create a lock using multiprocessing.Manager

    # process_video(root_directory, video_titles[i], transcript_titles[i], out_dir, log_file_path, 0, 0, lock, gpu_queue)
    # process_video(root_directory, video_titles[1], transcript_titles[1], out_dir, log_file_path, lock)

    # with Pool(processes=max_processes) as pool:
    #     results = []
    #     for video_title, transcript_title in zip(video_titles, transcript_titles):
    #         result = pool.apply_async(process_video, (root_directory, video_title, transcript_title, out_dir, log_file_path, lock), callback=lambda _: progress.update())
    #         results.append(result)
        
    #     for result in results:
    #         result.wait()

    for video_title, PID in zipped_titles:
        available_gpu, prev_process = gpu_queue.get()
        active_processes[available_gpu][prev_process] = None

        video_title = "P015_10062022_JF_4092"
        PID = "P015"

        # transcript_df = all_transcripts_df[all_transcripts_df["filename_stripped"] == video_title]
        # va_df = all_va_df[all_va_df["PID"] == PID]

        # process_video(root_directory, video_title, transcript_df, va_df, out_dir, log_file_path, available_gpu, prev_process, lock, gpu_queue, include_audio)
        # process_video(root_directory, video_title, transcript_df, va_df, roberta_df, roberta_red_df, out_dir, log_file_path, available_gpu, prev_process, lock, gpu_queue, include_audio)

        video_encoding_proc = Process(target=process_video, args=(root_directory, video_title, transcript_df, va_df, out_dir, log_file_path, available_gpu, prev_process, lock, gpu_queue, include_audio))
        video_encoding_proc.start()
        active_processes[available_gpu].append(video_encoding_proc)
    
    for proc_list in active_processes:
        for process in proc_list:
            if process is not None:
                process.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video frames and save as HDF5.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory path containing video-specific subdirectories.')
    parser.add_argument('--out_dir', type=str, required=True, help='TDirectory path to HDF5 files.')
    parser.add_argument('--transcript_file', type=str, required=True, help='File with transcript data.')
    parser.add_argument('--va_file', type=str, required=True, help='File with valence/arousal data.')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file.')
    parser.add_argument('--gpus', type=str, required=True, help='GPU IDs designating what GPUs are being used.')
    parser.add_argument('--tasks_per_gpu', type=str, required=True, help='How many processes per GPU to run')
    parser.add_argument('--exclude_audio', action='store_true', help='Exclude audio sampling')
    
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir
    transcript_file = args.transcript_file
    va_file = args.va_file
    log_file_path = args.log_file
    gpus = [int(x) for x in args.gpus.split(" ")]
    tasks_per_gpu = [int(x) for x in args.tasks_per_gpu.split(" ")]
    include_audio = not args.exclude_audio

    save_to_hdf5(input_dir, out_dir, transcript_file, va_file, log_file_path, gpus, tasks_per_gpu, include_audio)

