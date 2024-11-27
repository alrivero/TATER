import argparse
import h5py
import pandas as pd
from tqdm import tqdm
import json
import os

def explore_h5_file(file_path, csv_path, output_json):
    filename_id = file_path.split("/")[-1].split("_")[0]
        
    df = pd.read_csv(csv_path)  # Read the CSV
    filtered_df = df[df['user_id'] == filename_id]
    
    print(f"Processing HDF5 file: {file_path}")
    print(filtered_df.head())
    
    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r+') as f:
        # List all keys in the root of the HDF5 file
        keys = list(f.keys())
        file_og_ids = [
            int(f[key].attrs["og_id"])
            for key in keys 
            if "og_id" in f[key].attrs
        ]
        to_remain, to_add, interviewer = [], [], []
        for idx, row in tqdm(filtered_df.iterrows(), desc="Processing CSV rows"):
            t_id = int(row["message_id"].split("_")[-1])
            speaker = row["speaker_role"]
            if speaker == "participant":
                if t_id in file_og_ids:
                    to_remain.append(t_id)
                else:
                    to_add.append(t_id)

        to_remove = [id for id in file_og_ids if id not in to_remain]
        filtered_og_ids = [id for id in file_og_ids if id in to_remain]
        filtered_og_ids.extend(to_add)

    # Write to JSON
    if os.path.exists(output_json):
        # Load existing JSON if it exists
        with open(output_json, 'r') as json_file:
            output_data = json.load(json_file)
    else:
        output_data = {}

    # Add the current file's data
    output_data[file_path] = {
        "to_add": to_add,
        "to_remove": to_remove
    }

    # Save the updated JSON
    with open(output_json, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
    print(f"Updated JSON saved to: {output_json}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Explore HDF5 file and synchronize with a CSV.")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("output_json", type=str, help="Path to the output JSON file.")

    args = parser.parse_args()

    csv_path = "../../HiTOP_transcripts_v4.csv"

    # Call the function with the provided arguments
    explore_h5_file(args.file_path, csv_path, args.output_json)