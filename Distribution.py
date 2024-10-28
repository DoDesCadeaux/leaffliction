import os
import matplotlib.pyplot as plt
import numpy as np
from sys import argv


def path_sub_dir(directory: str) -> list:
    if directory.endswith('/'):
        directory = directory.removesuffix('/')
    sub_dir = [f"{directory}/{i}" for i in os.listdir(directory) if os.path.isdir(f"{directory}/{i}")]
    return sub_dir


def list_sub_dir(directory: str) -> list:
    return [i for i in os.listdir(directory) if os.path.isdir(f"{directory}/{i}")]


def get_sub_dir_sizes(sub_dirs: list, path_sub: list) -> dict:
    sub_dir_sizes = {}

    for n, folder in enumerate(sub_dirs):
        valid_images = [f for f in os.listdir(path_sub[n]) if f.lower().endswith(('.png', '.jpeg', '.jpg'))]
        sub_dir_sizes[folder] = len(valid_images)

    return sub_dir_sizes


if __name__ == "__main__":
    if len(argv) != 2:
        print("Incorrect number of args")
        exit(1)

    folder = argv[1]

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory")
        exit(1)

    sub_dirs = list_sub_dir(folder)

    if not sub_dirs:
        print(f"Error: No directories found in {sub_dirs}")
        exit(1)

    sub_dir_sizes = get_sub_dir_sizes(
        list_sub_dir(folder),
        path_sub_dir(folder)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(
        sub_dir_sizes.keys(),
        sub_dir_sizes.values(),
        color=plt.cm.Set1(np.linspace(0, 1, len(sub_dir_sizes))))

    ax1.set_ylabel("Leaves count")
    ax1.set_xlabel("Leaf type")
    ax1.set_xticklabels(sub_dir_sizes.keys(), rotation=90)

    fig.suptitle("Leaves Class Distribution")
    ax2.pie(
        sub_dir_sizes.values(),
        labels=sub_dir_sizes.keys(),
        autopct='%1.0f%%',
        colors=plt.cm.Set1(np.linspace(0, 1, len(sub_dir_sizes))))

    fig.tight_layout()
    plt.show()
