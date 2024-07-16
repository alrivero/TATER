import os
import h5py
import numpy as np
import torch
import cv2
import argparse
import pickle
import albumentations as A
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from skimage import transform as trans
from skimage.transform import estimate_transform, warp

# these are the indices of the mediapipe landmarks that correspond to the mediapipe landmark barycentric coordinates provided by FLAME2020
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

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
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
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
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

def prepare_data(image, landmarks_fan, landmarks_mediapipe, crop_res):
    # Find the median resolution of all frames
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    if landmarks_fan is None:
        flag_landmarks_fan = False
        landmarks_fan = np.zeros((68,2))
    else:
        flag_landmarks_fan = True
        if len(landmarks_fan.shape) == 3:
            landmarks_fan = landmarks_fan[0]

    tform = crop_face_final(image, landmarks_mediapipe, 1.8, image_size=crop_res)
    
    landmarks_mediapipe = landmarks_mediapipe[...,:2]
    
    cropped_image = warp(image, tform.inverse, output_shape=(crop_res, crop_res), preserve_range=True).astype(np.uint8)
    cropped_landmarks_fan = np.dot(tform.params, np.hstack([landmarks_fan, np.ones([landmarks_fan.shape[0],1])]).T).T
    cropped_landmarks_fan = cropped_landmarks_fan[:,:2]

    cropped_landmarks_mediapipe = np.dot(tform.params, np.hstack([landmarks_mediapipe, np.ones([landmarks_mediapipe.shape[0],1])]).T).T
    cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[:,:2]

    # find convex hull for masking the face 
    hull_mask = create_mask(cropped_landmarks_mediapipe, (crop_res, crop_res))

    cropped_landmarks_mediapipe = cropped_landmarks_mediapipe[mediapipe_indices,:2]

    cropped_landmarks_fan[:,:2] = cropped_landmarks_fan[:,:2]/crop_res * 2  - 1
    cropped_landmarks_mediapipe[:,:2] = cropped_landmarks_mediapipe[:,:2]/crop_res * 2  - 1


    # ----------- mica images ---------------- #
    landmarks_arcface_crop = landmarks_fan[[36,45,32,48,54]].copy()
    landmarks_arcface_crop[0] = (landmarks_fan[36] + landmarks_fan[39])/2
    landmarks_arcface_crop[1] = (landmarks_fan[42] + landmarks_fan[45])/2

    tform = estimate_norm(landmarks_arcface_crop, 112)

    image = image/255.
    mica_image = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)
    mica_image = mica_image.transpose(2,0,1)

    image = cropped_image
    landmarks_fan = cropped_landmarks_fan
    landmarks_mediapipe = cropped_landmarks_mediapipe
    hull_mask = hull_mask
    mica_image = mica_image


    data_dict = {
        'img': image[None],
        'landmarks_fan': landmarks_fan[...,:2][None],
        'flag_landmarks_fan': flag_landmarks_fan, # if landmarks are not available
        'landmarks_mp': landmarks_mediapipe[...,:2][None],
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
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    if landmarks_fan is None:
        landmarks_fan = np.zeros((68,2))
    else:
        if len(landmarks_fan.shape) == 3:
            landmarks_fan = landmarks_fan[0]

    _, new_size = crop_face(image, landmarks_mediapipe, 1.0)

    return new_size

def process_video(video_path):
    images_folder = [f for f in os.listdir(os.path.join(video_path, 'images'))]
    fan_lmks_folder = [f for f in os.listdir(os.path.join(video_path, 'fan_landmarks'))]
    mediapipe_lmks_folder = [f for f in os.listdir(os.path.join(video_path, 'mediapipe_landmarks'))]

    data_dict = {
        'img': [],
        'landmarks_fan': [],
        'flag_landmarks_fan': [], # if landmarks are not available
        'landmarks_mp': [],
        'mask': [],
        'img_mica': []
    }
    frames = []
    all_fan_landmarks = []
    all_mp_landmarks = []
    all_masks_landmarks = []
    
    crop_reses = []
    for img_file, fan_file, mp_file in zip(images_folder, fan_lmks_folder, mediapipe_lmks_folder):
        img_path = os.path.join(video_path, 'images', img_file)
        fan_path = os.path.join(video_path, 'fan_landmarks', fan_file)
        mp_path = os.path.join(video_path, 'mediapipe_landmarks', mp_file)

        frame = cv2.imread(img_path)

        with open(fan_path, "rb") as pkl_file:
            landmarks = np.load(pkl_file)

            preprocessed_landmarks = landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                raise Exception('Video %s has no landmarks'%(video_path))
        mediapipe_landmarks = np.load(mp_path)

        crop_reses.append(get_crop_res(frame, landmarks, mediapipe_landmarks))

        frames.append(frame)
        all_fan_landmarks.append(landmarks)
        all_mp_landmarks.append(mediapipe_landmarks)
    
    final_crop_res = int(np.median(np.array(crop_reses)))
    for frame, fan_lmks, mp_lmks in zip(frames, all_fan_landmarks, all_mp_landmarks):
        frame_data = prepare_data(frame, fan_lmks, mp_lmks, final_crop_res)

        for key in data_dict.keys():
            data_dict[key].append(frame_data[key])
    
    for key in data_dict.keys():
        if key != 'flag_landmarks_fan':
            data_dict[key] = np.concatenate(data_dict[key])

    return data_dict

def save_to_hdf5(root_directory, hdf5_path):
    with h5py.File(hdf5_path, 'w') as f:
        idx = 0
        for video_dir in sorted(os.listdir(root_directory)):
            video_path = os.path.join(root_directory, video_dir)

            print(f"Processing {video_path}")
            video_dict = process_video(video_path)
            group = f.create_group(f'{idx}')
            for key, value in video_dict.items():
                group.create_dataset(key, data=value)
            idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video frames and save as HDF5.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory path containing video-specific subdirectories.')
    parser.add_argument('--output_file', type=str, required=True, help='Output HDF5 file path.')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

    save_to_hdf5(input_dir, output_file)
