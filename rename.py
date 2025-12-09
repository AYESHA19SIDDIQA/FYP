import os
import re

def rename_fixation_files(directory):
    """
    Rename fixation files from 'fixations_0000004' format to '4_P0.json'
    """
    renamed_count = 0
    
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue
            
        # Check if it matches the pattern
        if filename.startswith('fixations_'):
            # Extract the number part
            match = re.search(r'fixations_(\d+)\.json', filename)
            if match:
                number_str = match.group(1)
                # Remove leading zeros
                number = int(number_str)
                # Create new filename
                new_filename = f"{number}_P0.json"
                
                # Get full paths
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                # Check if new filename already exists
                if os.path.exists(new_path):
                    print(f"Warning: {new_filename} already exists! Skipping {filename}")
                    continue
                
                try:
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"✓ Renamed: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"✗ Error renaming {filename}: {e}")
    
    return renamed_count

def main():
    # Directories to process
    directories = [
        r"E:\gaze_simulated\normal_fixation",
        r"E:\gaze_simulated\abnormal_fixation"
    ]
    
    total_renamed = 0
    
    # Process each directory
    for directory in directories:
        print(f"\n{'='*60}")
        print(f"Processing directory: {directory}")
        print(f"{'='*60}")
        
        if os.path.exists(directory):
            renamed = rename_fixation_files(directory)
            total_renamed += renamed
            print(f"\nRenamed {renamed} files in {directory}")
        else:
            print(f"✗ Directory not found: {directory}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: Renamed {total_renamed} files across all directories")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Ask for confirmation
    print("This script will rename JSON files from 'fixations_0000004.json' format to '4_P0.json'")
    print("Directories to process:")
    print("1. E:\\gaze_simulated\\normal_fixation")
    print("2. E:\\gaze_simulated\\abnormal_fixation")
    
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        main()
        print("\nOperation completed!")
    else:
        print("Operation cancelled.")