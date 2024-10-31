import random
import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from sys import argv
from Distribution import get_sub_dir_sizes, list_sub_dir, path_sub_dir


def flip(image: np.array, axis: str | int) -> np.array:
    accepted_axis = ['horizontal', 'vertical', 'h', 'v', 0, 1]

    if axis not in accepted_axis:
        print(f"Error: axis parameter has no attribute '{axis}'")
        exit(1)

    if axis in ['horizontal', 'h', 0]:
        return image[::-1, ::, ::-1]
    elif axis in ['vertical', 'v', 1]:
        return image[::, ::-1, ::-1]


def rotation(image: np.array, degrees: int) -> np.array:
    if degrees % 90 != 0:
        print("Error: only multiple of 90 for rotation is allowed (90, 180, ...)")
        exit(1)

    center = (image.shape[1] // 2, image.shape[0] // 2)
    scale = 1

    rotation_matrix = cv.getRotationMatrix2D(center, degrees, scale)

    rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image[:, :, ::-1]


def blur(image: np.array, blur_type: str) -> np.array:
    blur_types = ['gaussian', 'median', 'bilateral']

    if blur_type not in blur_types:
        print(f"Error: blur_type parameter ahs no attribute {blur_type}")
        exit(1)

    if blur_type == 'bilateral':
        return cv.bilateralFilter(image[:, :, ::-1], 9, 75, 75)
    elif blur_type == 'gaussian':
        return cv.GaussianBlur(image[:, :, ::-1], (7, 7), 0)
    return cv.medianBlur(image[:, :, ::-1], 5)


def color_filtering(image: np.array, red: float, green: float, blue: float) -> np.array:
    b, g, r = cv.split(image)

    b = (b * blue).clip(0, 255).astype(np.uint8)
    g = (g * green).clip(0, 255).astype(np.uint8)
    r = (r * red).clip(0, 255).astype(np.uint8)

    image_reduced = cv.merge([b, g, r])
    return image_reduced[:, :, ::-1]


def contrast(image: np.array, alpha: float, beta: float) -> np.array:
    new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)

    return new_image[:, :, ::-1]


def dilation(image: np.array) -> np.array:
    kernel = np.ones((2, 2), np.uint8)

    img_dilation = cv.dilate(image, kernel, iterations=1)
    return img_dilation[:, :, ::-1]


def scaling(image: np.array) -> np.array:
    desired_width = 256
    desired_height = 256

    cropped_img = image[25:, 35:220]

    dim = (desired_width, desired_height)
    resized_cropped = cv.resize(cropped_img, dsize=dim, interpolation=cv.INTER_AREA)

    return resized_cropped[:, :, ::-1]


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


def get_dir_size(path: str) -> int:
    files = os.listdir(path)

    return len(files)


def calculate_image_needed(sub_dirs):
    max_size = max(sub_dirs.values())

    images_needed = {}

    for folder, size in sub_dirs.items():
        difference = int(max_size) - int(size)
        images_needed[folder] = math.ceil(difference / 6)

    return images_needed


def augmentation(image: np.array, file_path: str, random_automatic: int) -> None:
    transformations = {
        '_Flip': flip(image, 1),
        '_Rotate': rotation(image, 90),
        '_Blur': blur(image, 'bilateral'),
        '_Crop': scaling(image),
        '_Dilation': dilation(image),
        '_Contrast': contrast(image, 1.5, 0)
    }

    split_path = file_path.split("/")
    folder_path = split_path[0]
    sub_dir_path = "/".join(split_path[1:-1])
    file_name, file_ext = split_path[-1].split(".")

    if random_automatic == 0:
        augmented_dir = os.path.join("augmented_directory", sub_dir_path)
        os.makedirs(augmented_dir, exist_ok=True)

        for transf_name, transformation in transformations.items():
            file_augment_dir = f"{augmented_dir}/{file_name}{transf_name}.{file_ext}"
            file_actual_dir = f"{folder_path}/{sub_dir_path}/{file_name}{transf_name}.{file_ext}"
            cv.imwrite(file_augment_dir, transformation)
            cv.imwrite(file_actual_dir, transformation)
    else:
        dir_sizes = get_sub_dir_sizes(list_sub_dir(folder_path), path_sub_dir(folder_path))

        images_needed = calculate_image_needed(dir_sizes)

        for sub_dir, old_length in dir_sizes.items():
            augmented_dir = os.path.join("augmented_directory", sub_dir)
            os.makedirs(augmented_dir, exist_ok=True)

            images = os.listdir(f"{folder_path}/{sub_dir}")
            random_images = random.sample(images, images_needed[sub_dir])

            if len(random_images) > 0:
                for transf_name, transformation in transformations.items():
                    for image in random_images:
                        file_name, file_ext = image.split(".")
                        file_augment_dir = f"{augmented_dir}/{file_name}{transf_name}.{file_ext}"
                        file_actual_dir = f"{folder_path}/{sub_dir}/{file_name}{transf_name}.{file_ext}"
                        cv.imwrite(file_augment_dir, transformation)
                        cv.imwrite(file_actual_dir, transformation)


def main():
    image_path = argv[1]

    img_read_bgr = cv.imread(image_path, cv.IMREAD_COLOR)
    img_read_rgb = cv.cvtColor(img_read_bgr, cv.COLOR_BGR2RGB)

    try:
        with open('data_augmentation.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        with open('data_augmentation.json', 'w') as outfile:
            data = {
                'already_augmented': 0
            }

            json.dump(data, outfile)

    if data['already_augmented'] == 0:
        augmentation(img_read_rgb, image_path, 1)

        data = {
            'already_augmented': 1
        }

        with open('data_augmentation.json', 'w') as file:
            json.dump(data, file)
    elif data['already_augmented'] == 1:
        print("Data Already Augmented")
        exit(1)


if __name__ == "__main__":
    main()
