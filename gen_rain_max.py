import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

# Function to find max rain files and save as rain_max.csv
def max_rain_files(root_dir='train_data'):
    # Check if root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return

    # Iterate through each event folder
    event_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Sort event_ids for consistent processing order (optional but good)
    event_ids.sort()

    for event_id in tqdm(event_ids, desc="Processing Events"):
        rain_dir = os.path.join(root_dir, event_id, 'rain')
        
        if not os.path.exists(rain_dir):
            continue

        # Get all CSV files in the rain directory
        all_files = os.listdir(rain_dir)
        
        # Filter for relevant csv files, excluding the target output file if it already exists
        rain_files = [f for f in all_files if f.endswith('.csv') and f not in ['rain.csv', 'rain_max.csv'] and not f.startswith('.')]
        
        if not rain_files:
            continue

        # Read and concatenate
        data_frames = []
        for file in rain_files:
            file_path = os.path.join(rain_dir, file)
            # Assuming files are headerless matrices as per standard ML dataset formats similar to dataload.py check
            # Using header=None. If your files have headers, change this.
            try:
                df = pd.read_csv(file_path, header=None)
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if data_frames:
            
            # Convert list of DataFrames to list of numpy arrays
            arrays = [df.values for df in data_frames]
            
            # Stack into 3D array: (Time, Height, Width)
            stacked_data = np.stack(arrays, axis=0) # Shape: (T, H, W)
            
            # Calculate Max across Time dimension (axis 0)
            max_grid = np.max(stacked_data, axis=0) # Shape: (H, W)
            
            output_path = os.path.join(rain_dir, 'rain_max.csv')
            
            # Save to CSV
            pd.DataFrame(max_grid).to_csv(output_path, header=False, index=False)
            
            # Optional: Print info for first event to verify
            # print(f"Merged {len(rain_files)} files in {event_id} into shape {combined_df.shape}")

if __name__ == "__main__":
    max_rain_files()
