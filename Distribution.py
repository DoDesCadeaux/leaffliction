import os
import matplotlib.pyplot as plt
import numpy as np
from sys import argv


def path_sub_dir(directory: str) -> list:
    sub_dir = os.listdir(directory)
    paths = []
    for i in sub_dir:
        paths.append(f"{directory}/{i}")

    return paths


def list_sub_dir(directory: str) -> list:
    sub_dir = os.listdir(directory)
    return sub_dir


def get_sub_dir_sizes(sub_dirs: list, path_sub: list) -> dict:
    sub_dir_sizes = {}

    for n, folder in enumerate(sub_dirs):
        sub_dir_sizes[folder] = len(list_sub_dir(path_sub[n]))

    return sub_dir_sizes


if __name__ == "__main__":
    if len(argv) != 2:
        print("Incorrect number of args")
        exit(1)

    folder = argv[1]

    sub_dir_sizes = get_sub_dir_sizes(
        list_sub_dir(folder),
        path_sub_dir(folder)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(
        sub_dir_sizes.keys(),
        sub_dir_sizes.values(),
        color=plt.cm.viridis(np.linspace(0, 1, len(sub_dir_sizes))))
    ax1.set_ylabel("Leaves count")
    ax1.set_xlabel("Leaf type")
    ax1.set_xticklabels(sub_dir_sizes.keys(), rotation=90)

    fig.suptitle("Leaves Class Distribution")
    ax2.pie(
        sub_dir_sizes.values(),
        labels=sub_dir_sizes.keys(),
        autopct='%1.0f%%')
    fig.tight_layout()
    plt.show()
