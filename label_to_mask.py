from PIL import Image
import numpy as np

import os

# Specify the directory you want to list
folder_path = 'D:\\github\\Segment-Any-Axon\\datasets\\data_axondeepseg_sem\\images'

# Get all files (excluding directories) in the specified directory
file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
              if os.path.isfile(os.path.join(folder_path, filename))]

folder_path = 'D:\\github\\Segment-Any-Axon\\datasets\\osfstorage-archive\\images'
file_paths.extend([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
              if os.path.isfile(os.path.join(folder_path, filename))])
folder_path = 'D:\\github\\Segment-Any-Axon\\datasets\\zenodo\\images'
file_paths.extend([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
              if os.path.isfile(os.path.join(folder_path, filename))])

destination_folder="images_tif"
for image_path in file_paths:
    # Load the image
    image = Image.open(image_path)
    
    # Extract the basename and create a new file name with .tif extension
    basename = os.path.basename(image_path)
    tiff_filename = os.path.splitext(basename)[0] + '.tif'
    
    # Specify the destination path
    tiff_path = os.path.join(destination_folder, tiff_filename)
    
    # Save the image as TIFF
    image.save(tiff_path, format='TIFF')
exit()
# Print out all file paths
for img_path in file_paths:
    img_name=os.path.basename(img_path)[:-4]
    image = Image.open(img_path).convert('L')  # Convert to grayscale

    # Convert image to a numpy array
    image_array = np.array(image)

    # Define thresholds for black, gray, and white
    # Assuming the following ranges based on grayscale values (0-255):
    # Black: 0-85, Gray: 86-170, White: 171-255
    black_threshold = 85
    white_threshold = 170

    # Initialize the segmentation mask with zeros
    segmentation_mask = np.zeros_like(image_array)

    # Assign 1 to gray pixels
    segmentation_mask[(image_array > black_threshold) & (image_array <= white_threshold)] = 1

    # Assign 2 to white pixels
    segmentation_mask[image_array > white_threshold] = 2

    # Count the unique values (0, 1, 2) and their counts
    unique, counts = np.unique(segmentation_mask, return_counts=True)
    unique_counts = dict(zip(unique, counts))

    # Show the segmentation mask and the counts of unique values
    # unique_counts, Image.fromarray(segmentation_mask * 127).show()  # Multiplied by 127 for visualization purposes

    mask_path = 'mask_tif/'+img_name+'.tif'
    Image.fromarray(segmentation_mask).save(mask_path, format='TIFF')