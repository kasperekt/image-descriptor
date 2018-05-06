import numpy as np
from skimage import exposure

from image import get_patch


class ExposureDescriptor:
    def __init__(self, patch_size=(16, 16)):
        self._patch_size = patch_size
        self._bins = 10

    def extract(self, image, keypoints):
        descriptors = [self.extract_for_keypoint(kp) for kp in keypoints]
        return descriptors

    def extract_for_keypoint(self, image, keypoint):
        patch = get_patch(image, keypoint, size=self._patch_size)
        histogram, _ = exposure.histogram(patch, nbins=self._bins)
        return histogram
