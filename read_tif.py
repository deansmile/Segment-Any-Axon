import tifffile as tiff

# Replace 'path_to_file.tif' with the path to your TIFF file
image = tiff.imread('D:\\github\\Segment-Any-Axon\\data\\instance_labeling\\01_0.tif')
print(image.min())
print(image.max())
exit()

image = tiff.imread('D:\\github\\Segment-Any-Axon\\data\\image_tif\\01_0.tif')
print(image.shape)

image = tiff.imread('D:\\github\\Segment-Any-Axon\\data\\mask_tif\\01_0.tif')
print(image.shape)