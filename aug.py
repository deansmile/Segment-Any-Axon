# Redefining the augmentation functions specifically for a 2D synthetic image
from skimage import transform, io, img_as_float
from scipy.ndimage import gaussian_filter, map_coordinates
import numpy as np
import matplotlib.pyplot as plt
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

image = img_as_float(io.imread("D:\\github\\Segment-Any-Axon\\zenodo\\Images\\7_EM1_3_10D_N_P_0002.jpg"))
# Apply the redefined augmentation functions to the synthetic image
augmented_images_2d = [
    random_shift_2d(image),
    random_rotation_2d(image),
    random_rescale_2d(image),
    random_flip_2d(image),
    random_blur_2d(image)
]

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


# Plot the original and augmented images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')

titles = ['Shifted', 'Rotated', 'Rescaled', 'Flipped', 'Blurred']
for ax, img, title in zip(axes[1:], augmented_images_2d, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)

plt.tight_layout()
plt.show()
