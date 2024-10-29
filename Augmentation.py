import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from sys import argv
from Distribution import get_sub_dir_sizes, list_sub_dir, path_sub_dir


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
        return cv.bilateralFilter(image, 9, 75, 75)
    elif blur_type == 'gaussian':
        return cv.GaussianBlur(image, (7, 7), 0)
    return cv.medianBlur(image, 5)


def color_filtering(image: np.array, red: float, green: float, blue: float) -> np.array:
    b, g, r = cv.split(image)

    b = (b * blue).clip(0, 255).astype(np.uint8)
    g = (g * green).clip(0, 255).astype(np.uint8)
    r = (r * red).clip(0, 255).astype(np.uint8)

    image_reduced = cv.merge([b, g, r])
    return image_reduced


def contrast(image: np.array, alpha: float, beta: float) -> np.array:
    new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)

    return new_image


def dilation(image: np.array) -> np.array:
    kernel = np.ones((2, 2), np.uint8)

    img_dilation = cv.dilate(image, kernel, iterations=1)
    return img_dilation


def scaling(image: np.array) -> np.array:
    desired_width = 256
    desired_height = 256

    cropped_img = image[25:, 35:220]

    dim = (desired_width, desired_height)
    resized_cropped = cv.resize(cropped_img, dsize=dim, interpolation=cv.INTER_AREA)

    return resized_cropped


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


def random_augmentation(folder: str) -> None:
    dir_sizes = get_sub_dir_sizes(list_sub_dir(argv[1]), path_sub_dir(argv[1]))

    largest_dir_size = max(dir_sizes.values())

    print(path_sub_dir(argv[1]))
    print(dir_sizes)

    for sub_dir, size in dir_sizes.items():
        difference = largest_dir_size - size
        if size < largest_dir_size:
            for i in range(difference):
                with open(f"{folder}/{sub_dir}/augmented_image_{i}.txt", 'w') as f:
                    f.write(f"test: {i}")


def augmentation(image: np.array, file_path: str) -> None:
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
    file_name, file_ext = split_path[-1].split(".")

    augmented_dir = os.path.join(folder_path, "augmented_directory")
    os.makedirs(augmented_dir, exist_ok=True)

    for transf_name, transformation in transformations.items():
        file = f"{augmented_dir}/{file_name}{transf_name}.{file_ext}"
        cv.imwrite(file, transformation)


# Todo -> Count all sub_dir_images to augment the underrepresented data
# Todo -> Augment only the sub_dir with less images than biggest sub_dir
# Todo -> Random augmentation for all sub_dirs


def main():
    image_path = argv[1]

    img_read_bgr = cv.imread(image_path, cv.IMREAD_COLOR)
    img_read_rgb = cv.cvtColor(img_read_bgr, cv.COLOR_BGR2RGB)

    augmentation(img_read_rgb, image_path)

    # flip_img = flip(img_read_rgb, 1)
    # rotate_img = rotation(img_read_rgb, 90)
    # blur_image = blur(img_read_rgb, 'bilateral')
    # cropped = scaling(img_read_rgb)
    # dilated = dilation(img_read_rgb)
    # contrasted = contrast(img_read_rgb, 1.5, 0)
    #
    # images = [img_read_rgb, flip_img, rotate_img, blur_image, cropped, dilated, contrasted]
    # categories = ['Original', 'Flip', 'Rotation', 'Blur', 'Scaled', 'Dilation', 'Contrast']

    # show_images(images, categories)

    # random_augmentation(argv[1])


if __name__ == "__main__":
    main()
