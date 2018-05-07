import matplotlib.pyplot as plt
from math import ceil
import numpy as np


def plot_images(images, maxcols=3, figsize=(20, 6), cmap='gray'):
    n_images = len(images)
    n_cols = n_images if n_images < maxcols else maxcols
    n_rows = ceil(n_images / n_cols)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    for ax, image in zip(axes.flat, images):
        ax.imshow(image, cmap=cmap)


def plot_with_keypoints(image, keypoints, markersize=3, cmap='gray'):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    ax.plot(keypoints[:, 0], keypoints[:, 1], '.r', markersize=markersize)
