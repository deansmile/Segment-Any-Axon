# import os
# folder_path = 'D:\\github\\Segment-Any-Axon\\data\\image_tif'

# # Get all files (excluding directories) in the specified directory
# file_paths = [filename for filename in os.listdir(folder_path)
#               if os.path.isfile(os.path.join(folder_path, filename))]

# folder_path = 'D:\\github\\Segment-Any-Axon\\data\\mask_tif'
# file_paths1=[filename for filename in os.listdir(folder_path)
#               if os.path.isfile(os.path.join(folder_path, filename))]
# # Convert lists to sets
# set1 = set(file_paths)
# set2 = set(file_paths1)

# # Find strings that are in set1 but not in set2, and vice versa
# unique_to_set1 = set1 - set2
# unique_to_set2 = set2 - set1

# # Check if there are any unique strings and print them
# if unique_to_set1:
#     print("Strings in list1 but not in list2:", unique_to_set1)
# if unique_to_set2:
#     print("Strings in list2 but not in list1:", unique_to_set2)

# exit()

import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2gray
def pad_image_to_shape(img, ref_shape):
    """
    Convert an image to grayscale (if it's color), pad it to the reference shape with zeros at the bottom and right if necessary.

    Parameters:
    img (numpy.ndarray): Input image.
    ref_shape (tuple): Reference image shape (height, width).

    Returns:
    numpy.ndarray: Padded grayscale image.
    """
    # Convert color images to grayscale
    if img.ndim == 3:
        img = rgb2gray(img)

    # Calculate the padding widths
    pad_height = max(0, ref_shape[0] - img.shape[0])
    pad_width = max(0, ref_shape[1] - img.shape[1])

    # Apply padding to the bottom and right. No padding to the top and left.
    padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    return padded_img


image_folder = 'D:\\github\\Segment-Any-Axon\\data\\image_tif'

images = os.listdir(image_folder)

# Define your reference shape (height, width, [channels])
ref_shape = (512, 512)  # For example, (512, 512) for grayscale or (512, 512, 3) for RGB

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    img = imread(img_path)
    
    # Check if the image shape matches the reference shape (excluding the channel dimension if necessary)
    if img.shape != ref_shape:
        print(f"Padding {img_name} from {img.shape} to {ref_shape}")
        # Pad the image
        padded_img = pad_image_to_shape(img, ref_shape)

        # Save the padded image
        output_path = os.path.join(image_folder, img_name)
        imsave(output_path, padded_img)
