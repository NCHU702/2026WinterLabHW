import os
import glob
import pandas as pd
import config

def extract_metadata(source_path, target_dir):
    """Extracts the first 6 lines as metadata and saves to metadata.txt"""
    target_meta_file = os.path.join(target_dir, "metadata.txt")
    
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 6:
            print(f"Warning: {os.path.basename(source_path)} has less than 6 lines, cannot extract metadata.")
            return False

        header = lines[:6]
        with open(target_meta_file, 'w', encoding='utf-8') as f_meta:
            f_meta.writelines(header)
        return True
    except Exception as e:
        print(f"Error extracting metadata from {source_path}: {e}")
        return False

def process_asc_file(source_path, target_dir):
    filename = os.path.basename(source_path)
    # Output as CSV
    filename_csv = os.path.splitext(filename)[0] + ".csv"
    target_data_file = os.path.join(target_dir, filename_csv)          
    
    try:
        # Check if already exists to skip? 
        # For now, overwrite
        
        # Read with pandas
        # sep=r'\s+' handles variable whitespace
        # skip first 6 lines (metadata)
        df = pd.read_csv(source_path, sep=r'\s+', header=None, skiprows=6)
        
        # Save to CSV (comma separated, no header, no index)
        df.to_csv(target_data_file, index=False, header=False, sep=',')
            
        return True
    except Exception as e:
        print(f"Error processing {source_path}: {e}")
        return False

def main():
    print(f"Starting Flood Data Processing...")
    print(f"Source: {config.RAW_FLOOD_DIR}")
    print(f"Target: {config.OUTPUT_DIR}")
    
    if not os.path.exists(config.RAW_FLOOD_DIR):
        print("ERROR: Raw flood directory does not exist. Check config.py")
        return

    for folder_num, target_id in config.FLOOD_FOLDER_MAPPING.items():
        source_dir = os.path.join(config.RAW_FLOOD_DIR, folder_num)
        target_dir = os.path.join(config.OUTPUT_DIR, target_id, "flood")
        
        if not os.path.exists(source_dir):
            print(f"Skipping {target_id} (Source Folder {folder_num} not found)")
            continue
            
        # Create target directory
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Find files files
        pattern_d = os.path.join(source_dir, "dm1d*.asc")
        files_to_process = glob.glob(pattern_d)
        
        # Also max file
        path_max = os.path.join(source_dir, "dm1maxd0.asc")
        if os.path.exists(path_max):
            files_to_process.append(path_max)
        
        if not files_to_process:
             print(f"No files found for {target_id} in {source_dir}")
             continue

        print(f"Processing {len(files_to_process)} files for {target_id} ({config.ID_TO_NAME_MAPPING.get(target_id, 'Unknown')})...")

        # Extract metadata from the first found file
        # We prefer dm1maxd0 if available, but any works
        first_file = files_to_process[0]
        extract_metadata(first_file, target_dir)
        
        count = 0
        for info_file in files_to_process:
            if process_asc_file(info_file, target_dir):
                count += 1
                
        print(f"Completed {target_id}: {count} files converted.")

if __name__ == "__main__":
    main()
