import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import shutil
import tempfile

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Process images/videos with MediaPipe.')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory path')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
parser.add_argument('--num_processes', type=int, default=16, help='Number of processes to use for processing')
args = parser.parse_args()

# Function to process an image
def process_image(image_file, output_file, vis_file, face_detector):
    image = cv2.imread(image_file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load and process the image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = face_detector.detect(mp_image)

    # Check for detected faces
    if not detection_result.face_landmarks:
        return

    # Extract face landmarks
    landmarks = detection_result.face_landmarks[0]
    landmarks_np = np.array([[landmark.x * mp_image.width, landmark.y * mp_image.height, landmark.z] for landmark in landmarks])

    # Save landmarks
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, landmarks_np)

    # # Save visualization if required
    # if vis_file:
    #     for landmark in landmarks_np:
    #         cv2.circle(image, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)
    #     os.makedirs(os.path.dirname(vis_file), exist_ok=True)
    #     cv2.imwrite(vis_file, image)

# Function to extract frames from a video
def extract_frames(video_file, temp_dir):
    cap = cv2.VideoCapture(video_file)
    frame_files = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = os.path.join(temp_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_file, frame)
        frame_files.append(frame_file)
        count += 1

    cap.release()
    return frame_files

# Function to process a video
def process_video(video_file, output_dir, face_detector):
    temp_dir = tempfile.mkdtemp()
    try:
        frame_files = extract_frames(video_file, temp_dir)
        for frame_file in frame_files:
            output_file = os.path.join(output_dir, os.path.basename(frame_file).replace('.jpg', '.npy'))
            vis_file = os.path.join(output_dir, 'vis', os.path.basename(frame_file))
            process_image(frame_file, output_file, vis_file, face_detector)
    finally:
        shutil.rmtree(temp_dir)
    
    print(f"FINISHED {video_file}")

# Function to process a file
def process_file(root, file_name, face_detector):
    input_path = os.path.join(root, file_name)
    rel_path = os.path.relpath(input_path, args.input_dir)
    output_dir = os.path.join(args.output_dir, os.path.splitext(rel_path)[0])

    if file_name.lower().endswith(('.mp4', '.avi')):
        process_video(input_path, output_dir, face_detector)

# Main processing function
def process_sample(args):
    root, file_name = args
    face_detector_options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='assets/face_landmarker.task'),output_face_blendshapes=False,output_facial_transformation_matrixes=False,num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(face_detector_options)

    process_file(root, file_name, face_detector)

if __name__ == '__main__':
    all_files = []

    for root, _, files in os.walk(args.input_dir):
        for file_name in files:
            if file_name.lower().endswith(('.mp4', '.avi')):
                all_files.append((root, file_name))

    with Pool(args.num_processes) as pool:
        list(tqdm(pool.imap(process_sample, all_files), total=len(all_files)))
