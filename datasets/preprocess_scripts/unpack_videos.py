import os
import cv2
import sys

def unpack_videos(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        # Construct full file path
        file_path = os.path.join(input_dir, filename)

        # Check if the file is a video file (basic check based on file extension)
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            # Create a subdirectory for this video
            video_name = os.path.splitext(filename)[0]
            video_dir = os.path.join(output_dir, video_name, "images")
            os.makedirs(video_dir, exist_ok=True)

            # Open the video file
            cap = cv2.VideoCapture(file_path)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save the frame as a PNG image
                frame_filename = os.path.join(video_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1

            # Release the video capture object
            cap.release()

            print(f"Extracted {frame_count} frames from {filename} into {video_dir}")

    print("All videos have been processed.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python unpack_videos.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    unpack_videos(input_directory, output_directory)
