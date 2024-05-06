import os
import random
# Specify the directory path
f=open("rand.txt")
selected_files=[]
for line in f:
    s=line.strip().split()
    selected_files.append(s[2])
# print(selected_files)
# print(len(selected_files))

import shutil

# Paths to the 'train' and 'val' directories
train_dir = 'data\\train'
val_dir = 'data\\val'

# Function to create a directory if it doesn't exist
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Get a list of subdirectories in the 'train' directory
subdirectories = ['myelin_mask_tif']

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
