import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def invert_coords(coords):
    return np.array([[c[1], c[0]] for c in coords])


def trim_coords_to_shape(shape, coords):
    height, width = shape
    filtered = filter(lambda c: c[0] <= width and c[1] <= height, coords)
    return np.array(list(filtered))


def plot_images(images, maxcols=3, figsize=(20, 6), cmap='gray'):
    n_images = len(images)
    n_cols = n_images if n_images < maxcols else maxcols
    n_rows = ceil(n_images / n_cols)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    for ax, image in zip(axes, images):
        ax.imshow(image, cmap=cmap)


def plot_with_keypoints(image, keypoints, markersize=3, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.plot(keypoints[:, 0], keypoints[:, 1], '.r', markersize=markersize)
    plt.show()
