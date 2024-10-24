import Distribution
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sys import argv


def flip(image: np.array, axis: str | int) -> np.array:
    accepted_axis = ['horizontal', 'vertical', 'h', 'v', 0, 1]

    if axis not in accepted_axis:
        print(f"Error: axis parameter has no attribute '{axis}'")
        exit(1)

    if axis in ['horizontal', 'h', 0]:
        return image[::-1]
    elif axis in ['vertical', 'v', 1]:
        return image[::, ::-1]


def rotation(image: np.array, degrees: int) -> np.array:
    if degrees % 90 != 0:
        print("Error: only multiple of 90 for rotation is allowed (90, 180, ...)")
        exit(1)

    center = (image.shape[1] // 2, image.shape[0] // 2)
    scale = 1

    rotation_matrix = cv.getRotationMatrix2D(center, degrees, scale)

    rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image


def blur(image: np.array, blur_type: str) -> np.array:
    blur_types = ['gaussian', 'median', 'bilateral']

    if blur_type not in blur_types:
        print(f"Error: blur_type parameter ahs no attribute {blur_type}")
        exit(1)

    if blur_type == 'bilateral':
        return cv.bilateralFilter(image, 9, 60, 30)
    elif blur_type == 'gaussian':
        return cv.GaussianBlur(image, (7, 7), 0)
    else:
        return cv.medianBlur(image, 5)


def color_filtering(image: np.array, red: float, green: float, blue: float) -> np.array:
    b, g, r = cv.split(image)

    b = (b * blue).clip(0, 255).astype(np.uint8)
    g = (g * green).clip(0, 255).astype(np.uint8)
    r = (r * red).clip(0, 255).astype(np.uint8)

    image_reduced = cv.merge([b, g, r])

    return image_reduced


def dilation(image: np.array) -> np.array:
    kernel = np.ones((2, 2), np.uint8)

    img_dilation = cv.dilate(image, kernel, iterations=1)
    return img_dilation


def show_images(images: list, categories: list) -> None:
    n_rows = 2
    n_cols = (len(images) + 1) // 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axs = axs.flatten()

    last = 0

    for i, (ax, image, category) in enumerate(zip(axs, images, categories)):
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(category)
        last = i

    for j in range(last + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image = argv[1]

    img_read_bgr = cv.imread(image, cv.IMREAD_COLOR)
    img_read_rgb = cv.cvtColor(img_read_bgr, cv.COLOR_BGR2RGB)

    flip_img = flip(img_read_rgb, 1)
    rotate_img = rotation(img_read_rgb, 90)
    blur_image = blur(img_read_rgb, 'bilateral')

    filtered = color_filtering(img_read_rgb, 1, 0, 1)
    dilated = dilation(img_read_rgb)

    images = [img_read_rgb, flip_img, rotate_img, blur_image, filtered, dilated, dilated]
    categories = ['Original', 'Flip', 'Rotation', 'Blur', 'Color Filter', 'Dilation', 'Dilation 2']

    show_images(images, categories)
