import os
import imageio
import numpy as np

def process_and_save_masks(source_dir, axon_dir, myelin_dir):
    # Ensure the output directories exist
    os.makedirs(axon_dir, exist_ok=True)
    os.makedirs(myelin_dir, exist_ok=True)
    
    # Iterate over every file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".tif"):
            # Construct the full file path
            file_path = os.path.join(source_dir, filename)
            
            # Read the mask file
            mask = imageio.imread(file_path)
            
            # Create masks based on the specified conditions
            axon_mask = (mask == 2).astype(np.uint8)  # Binary mask for axons
            myelin_mask = (mask == 1).astype(np.uint8)  # Binary mask for myelin
            
            # Save the masks to the respective directories
            imageio.imwrite(os.path.join(axon_dir, filename), axon_mask)
            imageio.imwrite(os.path.join(myelin_dir, filename), myelin_mask)
            
            print(f"Processed and saved masks for {filename}")

# Specify the directories
source_directory = '/scratch/ds5725/micro-sam/data/mask_tif'
axon_directory = '/scratch/ds5725/micro-sam/data/axon_mask_tif'
myelin_directory = '/scratch/ds5725/micro-sam/data/myelin_mask_tif'

# Process and save the masks
process_and_save_masks(source_directory, axon_directory, myelin_directory)
