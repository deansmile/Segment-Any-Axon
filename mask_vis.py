import imageio
import numpy as np
import matplotlib.pyplot as plt

# Load the image and the mask
image_path = '03_4_img.tif'  # Replace with your image path
mask_path = '03_4.tif'    # Replace with your mask path
image = imageio.imread(image_path)
mask = imageio.imread(mask_path)

# Check if the original image is grayscale and needs to be converted to RGB
if len(image.shape) == 2:  # If grayscale
    image_rgb = np.stack((image,) * 3, axis=-1)
elif len(image.shape) == 3 and image.shape[2] == 3:  # If already color
    image_rgb = image
else:
    raise ValueError("The image format is not supported.")

# Define the colors for myelin and axon
myelin_color = [0, 255, 0]  # Green color for myelin
axon_color = [255, 0, 0]    # Red color for axon

# Create color overlays
myelin_overlay = np.zeros_like(image_rgb)
axon_overlay = np.zeros_like(image_rgb)

# Apply the colors to the mask
myelin_overlay[mask == 1] = axon_color

# Combine the original image with the overlays
highlighted_image = np.maximum(np.maximum(image_rgb, myelin_overlay), axon_overlay)

# Display the result using matplotlib
plt.imshow(highlighted_image)
plt.axis('off')  # Hide axis labels
plt.show()
