import Distribution
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sys import argv


def flip(image: np.array, axis: str | int):
    accepted_axis = ['horizontal', 'vertical', 'h', 'v', 0, 1]

    if axis not in accepted_axis:
        print(f"Error: axis parameter ahs no attribute '{axis}'")
        exit(1)

    if axis in ['horizontal', 'h', 0]:
        return image[::-1]
    elif axis in ['vertical', 'v', 1]:
        return image[::, ::-1]


if __name__ == "__main__":
    image = argv[1]

    img_read_bgr = cv.imread(image, cv.IMREAD_COLOR)
    img_read_rgb = cv.cvtColor(img_read_bgr, cv.COLOR_BGR2RGB)

    flip_img = flip(img_read_rgb, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
    ax1.imshow(img_read_rgb)
    ax2.imshow(flip_img)
    plt.show()
