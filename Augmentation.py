import Distribution
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sys import argv


def flip(image: np.array, axis: str | int):
    accepted_axis = ['horizontal', 'vertical', 'h', 'v', 0, 1]

    if axis not in accepted_axis:
        print(f"Error: axis parameter has no attribute '{axis}'")
        exit(1)

    if axis in ['horizontal', 'h', 0]:
        return image[::-1]
    elif axis in ['vertical', 'v', 1]:
        return image[::, ::-1]


def rotation(image: np.array, degrees: int):
    if degrees % 90 != 0:
        print("Error: only multiple of 90 for rotation is allowed (90, 180, ...)")
        exit(1)

    center = (image.shape[1] // 2, image.shape[0] // 2)
    scale = 1

    rotation_matrix = cv.getRotationMatrix2D(center, degrees, scale)

    rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image


def show_images(images: list):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 10))
    for ax, image in zip(axs, images):
        ax.imshow(image)
        ax.axis('off')
    plt.show()


if __name__ == "__main__":
    image = argv[1]

    img_read_bgr = cv.imread(image, cv.IMREAD_COLOR)
    img_read_rgb = cv.cvtColor(img_read_bgr, cv.COLOR_BGR2RGB)

    flip_img = flip(img_read_rgb, 0)
    rotate_img = rotation(img_read_rgb, 90)

    images = [img_read_rgb, flip_img, rotate_img]

    show_images(images)

