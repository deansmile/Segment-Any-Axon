import cv2
import numpy as np

from pathlib import Path

pathlist = Path('datasets/data_axondeepseg_sem/labels').glob('*')
for path in pathlist:
    # used for combining axon and myelin mask into one in the osfstorage dataset
    avf = cv2.imread('datasets/osfstorage-archive/labels/sample8/avf.png', 0)
    myelin = cv2.imread('datasets/osfstorage-archive/labels/sample8/myelin.png', 0)
    arows, acols = avf.shape
    new_image = [[0 for i in range(acols)] for j in range(arows)]

    for i in range(arows):
        for j in range(acols):
            if avf[i, j] == 255:
                new_image[i][j] = 255
            elif myelin[i, j] == 255:
                new_image[i][j] = 127

    cv2.imwrite(path, np.array(new_image))
