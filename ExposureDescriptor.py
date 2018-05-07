import numpy as np
from skimage import exposure

from image import get_patch


class ExposureDescriptor:
    def __init__(self, patch_size=(32, 32)):
        self._patch_size = patch_size
        self._bins = 10

    def extract(self, image, keypoints):
        descriptors = [self.extract_for_keypoint(
            image, kp) for kp in keypoints]
        return np.array(descriptors)

    def extract_for_keypoint(self, image, keypoint):
        patch = get_patch(image, keypoint, size=self._patch_size)
        histogram, _ = exposure.histogram(patch, nbins=self._bins)
        return histogram

    # TODO: better distance algorithm
    def distance(self, desc1, desc2):
        smaller_desc = min([desc1, desc2], key=lambda arr: len(arr))
        bigger_desc = max([desc1, desc2], key=lambda arr: len(arr))

        result_matrix = np.ones((len(smaller_desc), len(bigger_desc)))

        for i, hist1 in enumerate(smaller_desc):
            for j, hist2 in enumerate(bigger_desc):
                result_matrix[i, j] = np.abs(
                    hist1 - hist2).sum() / (self._bins * 256)

        return result_matrix.min(axis=0).mean()
