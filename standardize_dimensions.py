# this script is to make the label images of deepaxonseg 2D like in other images

import cv2
import numpy as np

from pathlib import Path

pathlist = Path('datasets/data_axondeepseg_sem/labels').glob('*')
for path in pathlist:
    file = cv2.imread(str(path))
    rows, columns, _ = file.shape
    new_image = np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
            pixel = tuple(file[i, j])
            if pixel == (127, 127, 127) or pixel == (128, 128, 128):
                new_image[i, j] = 127
            elif pixel == (255, 255, 255):
                new_image[i, j] = 255
    cv2.imwrite(str(path), new_image)
