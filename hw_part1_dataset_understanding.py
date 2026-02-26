#!/usr/bin/env python3
"""
HW Part 1 — Dataset Understanding

Dataset structure:

real_objects/
    object_folder_000/
        000000-color.jpg
        ...
        000008-color.jpg
        name.txt
    object_folder_001/
        ...

Each folder = one class.
name.txt contains the class name.

------------------------------------------------------
Your tasks:

1) Print:
   - number of classes
   - total number of images
   - number of images per class (min / max / mean)

2) Create a mapping:
   class_id -> class_name

3) Visualize:
   - 5 random training images
   - show image + class name as title

You must implement the TODO sections below.
------------------------------------------------------
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


IMG_SUFFIX = "-color.jpg"


# --------------------------------------------------
# TODO 1: Index the dataset
# --------------------------------------------------
def index_dataset(root: Path):
    """
    Return:
        classes: list of dicts with keys:
            {
                "id": int,
                "name": str,
                "images": list[Path]
            }
    """

    classes = []

    # TODO:
    # 1. Iterate over all subfolders of root (sorted order).
    # 2. Read name.txt (fallback to folder name if missing).
    # 3. Collect all *-color.jpg images.
    # 4. Assign class_id starting from 0.
    # 5. Append dictionary to classes list.

    # ----- YOUR CODE HERE -----
    for class_id, folder in enumerate(sorted(root.iterdir())):
        if not folder.is_dir():
            continue

        name_file = folder / "name.txt"
        if name_file.exists():
            class_name = name_file.read_text().strip()
        else:
            class_name = folder.name

        images = sorted(folder.glob(f"*{IMG_SUFFIX}"))

        classes.append({
            "id": class_id,
            "name": class_name,
            "images": images
        }
    )
    # --------------------------

    return classes


# --------------------------------------------------
# TODO 2: Print dataset statistics
# --------------------------------------------------
def print_dataset_stats(classes):
    """
    Print:
        - number of classes
        - total images
        - min / max / mean images per class
    """

    # ----- YOUR CODE HERE -----
    num_classes = len(classes)

    # count images per class
    counts = [len(c["images"]) for c in classes]
    total_images = int(np.sum(counts))

    min_imgs = int(np.min(counts)) if counts else 0
    max_imgs = int(np.max(counts)) if counts else 0
    mean_imgs = float(np.mean(counts)) if counts else 0.0

    print(f"Number of classes: {num_classes}")
    print(f"Total number of images: {total_images}")
    print(f"Images per class: min={min_imgs}, max={max_imgs}, mean={mean_imgs:.2f}")
    # --------------------------


# --------------------------------------------------
# TODO 3: Visualize random samples
# --------------------------------------------------
def visualize_random_samples(classes, num_samples=5):
    """
    Randomly select num_samples images from the dataset
    and display them with class name as title.
    """

    # Flatten all images into a single list:
    # [(img_path, class_name), ...]

    # ----- YOUR CODE HERE -----
    # flatten all images
    all_images = []
    for c in classes:
        for img in c["images"]:
            all_images.append((img, c["name"]))

    # random sample
    samples = random.sample(all_images, num_samples)

    # visualize
    plt.figure(figsize=(15, 3))

    for i, (img_path, class_name) in enumerate(samples):
        img = Image.open(img_path)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    # --------------------------


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help='Path to "real_objects" folder')
    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise RuntimeError(f"Root folder does not exist: {root}")

    # 1. Index dataset
    classes = index_dataset(root)

    # 2. Print statistics
    print_dataset_stats(classes)

    # 3. Visualize samples
    visualize_random_samples(classes, num_samples=5)


if __name__ == "__main__":
    main()
