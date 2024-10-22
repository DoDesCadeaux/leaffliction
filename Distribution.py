import os
import matplotlib.pyplot as plt
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

    sub_dir_sizes = get_sub_dir_sizes(list_sub_dir(argv[1]), path_sub_dir(argv[1]))
    print(sub_dir_sizes)


