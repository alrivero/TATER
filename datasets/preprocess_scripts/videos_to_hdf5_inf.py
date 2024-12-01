import pickle
import os
import threading
from queue import Queue
import h5py
import numpy as np
import ffmpeg
import argparse
import pandas as pd

# Function to extract frames using FFmpeg
def extract_frames(video_path):
    try:
        # Use ffmpeg.probe to retrieve video metadata
        probe = ffmpeg.probe(video_path)
        video_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
        fps = eval(video_stream["r_frame_rate"])
        width = int(video_stream["width"])
        height = int(video_stream["height"])

        # Decode video frames into raw RGB data
        out, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )
        frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return frames, fps
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return None, None

# Function to extract audio using FFmpeg
def extract_audio(video_path):
    try:
        # Extract audio as raw PCM data
        out, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="s16le", acodec="pcm_s16le")
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.int16)
        probe = ffmpeg.probe(video_path)
        audio_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
        sample_rate = int(audio_stream["sample_rate"])
        return audio, sample_rate
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None, None

# Function to process the selected video and add the results to the queue
def process_video(queue, video_path, dataset_name):
    try:
        print(f"Processing video: {video_path}")
        
        # Extract video frames and FPS
        video_array, fps = extract_frames(video_path)
        if video_array is None or fps is None:
            print(f"Skipping invalid video: {video_path}")
            return

        # Extract audio and sample rate
        audio_array, sample_rate = extract_audio(video_path)
        if audio_array is None or sample_rate is None:
            print(f"Skipping video due to audio issues: {video_path}")
            return

        queue.put((dataset_name, video_array, fps, audio_array, sample_rate))
        print(f"Processed and queued: {dataset_name}")

    except Exception as e:
        print(f"Unexpected error processing video {video_path}: {e}")

# Function to handle HDF5 file writes
def hdf5_writer(queue, output_file, metadata_dict):
    with h5py.File(output_file, "w") as h5file:
        while True:
            data = queue.get()
            if data is None:  # Stop signal
                print("Writer thread: Stopping.")
                break
            dataset_name, video_array, fps, audio_array, sample_rate = data
            
            try:
                # Create a group for the video
                video_group = h5file.create_group(dataset_name)
                
                # Add the video frames as a dataset
                video_group.create_dataset("frames", data=video_array)

                # Add the audio as a dataset
                video_group.create_dataset("audio", data=audio_array)

                # Add metadata attributes
                video_group.attrs["fps"] = fps
                video_group.attrs["sample_rate"] = sample_rate
                if dataset_name in metadata_dict:
                    metadata = metadata_dict[dataset_name]
                    for key, value in metadata.items():
                        video_group.attrs[key] = value
                
                print(f"Writer thread: Wrote dataset {dataset_name}")

            except Exception as e:
                print(f"Error writing dataset {dataset_name} to HDF5 file: {e}")

# Load metadata from CSV
def load_metadata(csv_path):
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        metadata_dict = {}
        
        for _, row in df.iterrows():
            # Construct the diaM_uttN format
            dataset_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            
            # Store metadata
            metadata_dict[dataset_name] = {
                "Speaker": row["Speaker"],
                "Emotion": row["Emotion"],
                "Sentiment": row["Sentiment"],
                "Dialogue_ID": row["Dialogue_ID"],
                "Utterance_ID": row["Utterance_ID"],
            }
        return metadata_dict
    except Exception as e:
        print(f"Error loading metadata from CSV file: {e}")
        return {}

# Main function to process the input directory
def main(input_dir, output_file, csv_file, num_threads):
    # Load metadata from the CSV file
    metadata_dict = load_metadata(csv_file)
    if not metadata_dict:
        print("Metadata could not be loaded. Exiting.")
        return

    # Create a thread-safe queue
    queue = Queue()

    # Collect all pycrop/pywork folders and process videos
    selected_videos = []
    dataset_names = []
    for subdir, dirs, files in os.walk(input_dir):
        if "pycrop" in subdir:
            parent_dir = os.path.dirname(subdir)
            pywork_dir = os.path.join(parent_dir, "pywork")
            score_file = os.path.join(pywork_dir, "scores.pckl")

            # Get all .avi files in the pycrop directory
            avi_files = sorted([file for file in os.listdir(subdir) if file.endswith(".avi")])

            if len(avi_files) > 1:
                # If there are multiple .avi files, load and process the scores.pckl
                if os.path.exists(score_file):
                    try:
                        # Load the score array
                        with open(score_file, "rb") as f:
                            scores = pickle.load(f)

                        if not scores or len(scores) != len(avi_files):
                            print(f"Score mismatch in: {score_file}")
                            continue

                        # Find the video with the highest mean score
                        mean_scores = [np.mean(score) for score in scores]
                        highest_index = np.argmax(mean_scores)
                        highest_video = avi_files[highest_index]
                    except Exception as e:
                        print(f"Error processing scores.pckl in {pywork_dir}: {e}")
                        continue
                else:
                    print(f"Scores file missing for: {subdir}")
                    continue
            elif len(avi_files) == 1:
                # If only one .avi file exists, use it directly
                highest_video = avi_files[0]
            else:
                print(f"No .avi files found in {subdir}")
                continue

            # Add the selected video to the processing list
            video_path = os.path.join(subdir, highest_video)
            dataset_name = f"{os.path.basename(parent_dir)}_{os.path.splitext(highest_video)[0]}"
            selected_videos.append(video_path)
            dataset_names.append(dataset_name)

    print(f"Found {len(selected_videos)} videos to process.")

    # Create and start the writer thread
    writer_thread = threading.Thread(target=hdf5_writer, args=(queue, output_file, metadata_dict))
    writer_thread.start()

    # Create worker threads to process videos
    threads = []
    for video_path, dataset_name in zip(selected_videos, dataset_names):
        t = threading.Thread(target=process_video, args=(queue, video_path, dataset_name))
        threads.append(t)
        t.start()

        # Ensure we don't exceed the specified number of threads
        if len(threads) >= num_threads:
            for t in threads:
                t.join()
            threads = []

    # Wait for any remaining threads to complete
    for t in threads:
        t.join()

    # Signal the writer thread to stop
    queue.put(None)
    writer_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos in subdirectories and save to an HDF5 file.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing subdirectories with videos.")
    parser.add_argument("output_file", type=str, help="Path to the output HDF5 file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing metadata.")
    parser.add_argument("num_threads", type=int, help="Number of threads to use for processing.")

    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.csv_file, args.num_threads)

