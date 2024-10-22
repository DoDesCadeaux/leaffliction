import os
import matplotlib.pyplot as plt
import numpy as np
from sys import argv

from distributed.diagnostics.progress_stream import colors
from networkx.algorithms.bipartite.basic import color


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

    sub_dir_sizes = get_sub_dir_sizes(list_sub_dir(folder), path_sub_dir(folder))

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.bar(sub_dir_sizes.keys(), sub_dir_sizes.values(), color=plt.cm.viridis(np.linspace(0, 1, len(sub_dir_sizes))))
    plt.ylabel("Leaves count")
    plt.xlabel("Leaf type")
    plt.xticks(rotation=90)

    plt.subplot(122)
    plt.pie(sub_dir_sizes.values(), labels=sub_dir_sizes.keys(), autopct='%1.0f%%')
    plt.tight_layout()
    plt.show()
