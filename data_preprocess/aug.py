# Redefining the augmentation functions specifically for a 2D synthetic image
from skimage import transform, io, img_as_float, color
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile as tiff
from PIL import Image
from skimage.transform import resize

def random_shift_2d(image):
    max_shift = 0.1 * np.array(image.shape)
    shift_values = np.round(np.random.uniform(-max_shift, max_shift))
    shifted_image = np.roll(image, shift_values.astype(int), axis=(0,1))
    return shifted_image

def random_rotation_2d(image):
    angle = np.random.uniform(5, 89)
    return transform.rotate(image, angle, mode='edge')

def random_rescale_2d(image):
    scale = np.random.uniform(1/1.2, 1.2)
    height, width = image.shape
    rescaled_image = transform.resize(image, (int(height*scale), int(width*scale)), mode='edge')
    if scale > 1:
        cropx = (rescaled_image.shape[0] - height) // 2
        cropy = (rescaled_image.shape[1] - width) // 2
        rescaled_image = rescaled_image[cropx:cropx+height, cropy:cropy+width]
    elif scale < 1:
        pad_width = ((height - rescaled_image.shape[0]) // 2, (width - rescaled_image.shape[1]) // 2)
        rescaled_image = np.pad(rescaled_image, pad_width, mode='constant', constant_values=0)
    return rescaled_image

def random_flip_2d(image):
    if np.random.rand() > 0.5:
        return np.fliplr(image)
    else:
        return np.flipud(image)

def random_blur_2d(image):
    sigma = np.random.uniform(0, 4)
    return gaussian_filter(image, sigma=sigma)

# image = img_as_float(io.imread("D:\\github\\Segment-Any-Axon\\zenodo\\images\\7_EM1_3_10D_N_P_0002.jpg"))
# # Apply the redefined augmentation functions to the synthetic image
# augmented_images_2d = [
#     random_shift_2d(image),
#     random_rotation_2d(image),
#     random_rescale_2d(image),
#     random_flip_2d(image),
#     random_blur_2d(image)
# ]

def elastic_deformation_2d(image, alpha, sigma):
    assert len(image.shape) == 2, "Image must be 2D"
    
    random_state = np.random.RandomState(None)

    shape = image.shape
    
    # Generate displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # Apply displacement
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    return distorted_image

folder_path = 'D:\\github\\Segment-Any-Axon\\datasets\\data_axondeepseg_sem\\labels'

# Get all files (excluding directories) in the specified directory
file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
              if os.path.isfile(os.path.join(folder_path, filename))]

folder_path = 'D:\\github\\Segment-Any-Axon\\datasets\\osfstorage-archive\\labels'
file_paths.extend([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
              if os.path.isfile(os.path.join(folder_path, filename))])
folder_path = 'D:\\github\\Segment-Any-Axon\\datasets\\zenodo\\labels'
file_paths.extend([os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
              if os.path.isfile(os.path.join(folder_path, filename))])

augmentation_functions = [
    random_shift_2d,
    random_rotation_2d,
    random_rescale_2d,
    random_flip_2d,
    random_blur_2d,
    elastic_deformation_2d  # Assuming alpha and sigma are predefined
]

def mask(image_array):
    black_threshold = 85
    white_threshold = 170

    # Initialize the segmentation mask with zeros
    segmentation_mask = np.zeros_like(image_array)

    # Assign 1 to gray pixels
    segmentation_mask[(image_array > black_threshold) & (image_array <= white_threshold)] = 1

    # Assign 2 to white pixels
    segmentation_mask[image_array > white_threshold] = 2
    return segmentation_mask

for image_path in file_paths:
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    image = io.imread(image_path)
    image_resized = resize(image, (512, 512), anti_aliasing=True)

    # Ensure image is in float format and rescale intensity to [0, 255] for mask computation
    image_float = img_as_float(image_resized) * 255

    # Apply the mask function
    segmentation_mask = mask(image_float).astype(np.int8)

    # Save the mask in TIFF format
    mask_path =  f"D:\\github\\Segment-Any-Axon\\data\\mask_tif\\{img_name}.tif"
    tiff.imwrite(mask_path, segmentation_mask)
    # Check if the image has more than 2 dimensions (e.g., RGB) and convert to grayscale if needed
    # if image.ndim == 3 and image.shape[2] != 3:
    #     print("Unexpected image shape, expected RGB or grayscale.")
    #     continue
    #     # Handle this situation according to your needs. This could be an error, or you might choose a specific channel, etc.
    # elif image.ndim == 3:
    #     # If it's a 3-channel image, convert to grayscale
    #     image = color.rgb2gray(image)
    # elif image.ndim != 2:
    #     print("Unexpected image dimensions.")
    #     continue
    
    image_float = img_as_float(image_resized)
    # Apply each augmentation and save the resulting image
    for i, aug_func in enumerate(augmentation_functions):
        # For elastic deformation, you need to specify alpha and sigma
        if aug_func == elastic_deformation_2d:
            augmented_image = aug_func( image_float, alpha=1000, sigma=30)  # Example values
        else:
            augmented_image = aug_func( image_float)
        augmented_image=augmented_image*255

        # Save the augmented image as a TIFF, using the basename
        tiff_output_path = f"D:\\github\\Segment-Any-Axon\\data\\mask_tif\\{img_name}_{i}.tif"  # Define your output path
        tiff.imwrite(tiff_output_path, mask(augmented_image).astype(np.uint8))

# Plot the original and augmented images
# fig, axes = plt.subplots(2, 3, figsize=(12, 8))
# axes = axes.ravel()
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Original')
#
# titles = ['Shifted', 'Rotated', 'Rescaled', 'Flipped', 'Blurred']
# for ax, img, title in zip(axes[1:], augmented_images_2d, titles):
#     ax.imshow(img, cmap='gray')
#     ax.set_title(title)
#
# plt.tight_layout()
# plt.show()
