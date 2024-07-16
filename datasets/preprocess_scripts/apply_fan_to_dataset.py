from tqdm import tqdm
import numpy as np
import os
import cv2
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
import argparse
from ibug.face_alignment.utils import plot_landmarks

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Process images/videos with https://github.com/hhj1897/face_alignment.')
parser.add_argument('--dir', type=str, required=True, help='Directory path containing video subdirectories with images')
args = parser.parse_args()

all_files = []

# Collect all PNG files from the images subdirectories
for video_dir in os.listdir(args.dir):
    images_dir = os.path.join(args.dir, video_dir, 'images')
    if os.path.isdir(images_dir):
        for file_name in os.listdir(images_dir):
            if file_name.lower().endswith('.png'):
                all_files.append((images_dir, file_name))

# Create a RetinaFace detector using Resnet50 backbone, with the confidence threshold set to 0.8
face_detector = RetinaFacePredictor(
    threshold=0.8, device='cuda:0',
    model=RetinaFacePredictor.get_model('mobilenet0.25')
)

# Create a facial landmark detector
landmark_detector = FANPredictor(
    device='cuda:0', model=FANPredictor.get_model('2dfan2_alt')
)

for root, file_name in tqdm(all_files):
    input_path = os.path.join(root, file_name)
    rel_path = os.path.relpath(input_path, args.dir)
    video_dir = os.path.join(args.dir, os.path.dirname(os.path.dirname(rel_path)))

    output_path = os.path.join(video_dir, 'fan_landmarks', os.path.splitext(file_name)[0] + '.npy')
    vis_path = os.path.join(video_dir, 'fan_landmarks_vis', file_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)

    image = cv2.imread(input_path)

    detected_faces = face_detector(image, rgb=False)

    landmarks, scores = landmark_detector(image, detected_faces, rgb=False)

    np.save(output_path, landmarks)

    for lmks, scs in zip(landmarks, scores):
        plot_landmarks(image, lmks, scs, threshold=0.2)

    cv2.imwrite(vis_path, image)
