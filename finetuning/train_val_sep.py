import os
import random
# Specify the directory path
directory_path = '/scratch/ds5725/micro-sam/data/train/axon_mask_tif'

# Get a list of file names in the directory
file_names = os.listdir(directory_path)

# If you only want files (excluding directories)
file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

sorted_file_names = sorted(file_names)

selected_files = []

# Select one file randomly from every group of seven
for i in range(0, len(sorted_file_names), 7):
    group = sorted_file_names[i:i+7]
    if group:  # Check if the group is not empty
        selected_file = random.choice(group)
        selected_files.append(selected_file)

# print(selected_files)
# print(len(selected_files))

import shutil

# Paths to the 'train' and 'val' directories
train_dir = '/scratch/ds5725/micro-sam/data/train'
val_dir = '/scratch/ds5725/micro-sam/data/val'

# Function to create a directory if it doesn't exist
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Get a list of subdirectories in the 'train' directory
subdirectories = next(os.walk(train_dir))[1]

# Iterate over each subdirectory
for subdir in subdirectories:
    # Define the train and val subdirectory paths
    train_subdir_path = os.path.join(train_dir, subdir)
    val_subdir_path = os.path.join(val_dir, subdir)

    # Create the subdirectory under 'val' if it doesn't exist
    create_dir_if_not_exists(val_subdir_path)
    
    for file_name in selected_files:
        # Construct the full paths to the source file and destination
        src_file_path = os.path.join(train_subdir_path, file_name)
        dst_file_path = os.path.join(val_subdir_path, file_name)
        
        # Move the file
        shutil.move(src_file_path, dst_file_path)
        
        print(f"Moved file {file_name} from {train_subdir_path} to {val_subdir_path}")

print("File transfer from 'train' to 'val' subdirectories is complete.")
