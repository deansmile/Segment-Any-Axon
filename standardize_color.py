# this is used to standardize the color of axon and myelin labels in the zenodo dataset
import cv2
import numpy as np

from pathlib import Path


def find_closest_color(pixel):
    pixel = np.array(pixel)
    myelin_color = np.array((51, 50, 203))  # blue
    axon_color1 = np.array((204, 46, 45))  # red
    axon_color2 = np.array((221, 220, 50))  # yellow

    axon1_distance = np.linalg.norm(pixel - axon_color1)
    axon2_distance = np.linalg.norm(pixel - axon_color2)
    myelin_distance = np.linalg.norm(pixel - myelin_color)
    black_distance = np.linalg.norm(pixel)

    distance_lst = [axon1_distance, axon2_distance, myelin_distance, black_distance]

    return distance_lst.index(min(distance_lst))


pathlist = Path('datasets/zenodo/labels').glob('*')
for path in pathlist:
    file = cv2.imread(str(path))
    rows, columns, _ = file.shape
    new_image = np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
            pixel = tuple(file[i, j])
            nearest_color = find_closest_color(pixel)
            if nearest_color <= 1:
                new_image[i, j] = 127
            elif nearest_color == 2:
                new_image[i, j] = 255
    cv2.imwrite(str(path), new_image)
