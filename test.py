import numpy as np
import os

from skimage import img_as_float
from skimage.io import imread
from skimage.exposure import histogram
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance

from generate_patches import read_labels
from descriptor import extract_for_patch, distance


def calculate_distance(ref_patch_path, original_patch_path):
    ref_patch = imread(ref_patch_path, as_grey=True)
    original_patch = imread(original_patch_path, as_grey=True)

    ref_descriptor = extract_for_patch(ref_patch)
    original_descriptor = extract_for_patch(original_patch)

    return distance(ref_descriptor, original_descriptor)


def main():
    labels = read_labels('./patches/viewpoint-1')
    wrong_labels = read_labels('./patches/viewpoint-2')
    wrong_zipped = zip(labels.values(), wrong_labels.values())

    scores = np.array([calculate_distance(r_path, o_path)
                       for r_path, o_path in labels.items()])
    wrong_scores = np.array([calculate_distance(r_path, w_path)
                             for r_path, w_path in wrong_zipped])
    merged_scores = np.concatenate((scores, wrong_scores))

    distances = np.zeros_like(scores)
    wrong_distances = np.ones_like(wrong_scores)
    merged_distances = np.concatenate((distances, wrong_distances))

    print(roc_auc_score(merged_distances, merged_scores))


if __name__ == '__main__':
    main()
