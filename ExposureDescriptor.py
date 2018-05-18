import numpy as np

from image import get_patch

from skimage import img_as_float, exposure
from scipy.spatial.distance import cosine, braycurtis


class ExposureDescriptor:
    def __init__(self, patch_size=(32, 32), threshold=None):
        self._patch_size = patch_size
        self._bins = 10
        self._thres = threshold

    def extract(self, image, keypoints):
        descriptors = [self.extract_for_keypoint(
            image, kp) for kp in keypoints]
        return np.array(descriptors)

    def extract_for_keypoint(self, image, keypoint):
        patch = get_patch(image, keypoint, size=self._patch_size)
        return self.extract_for_patch(patch)

    def extract_for_patch(self, patch):
        histogram, _ = exposure.histogram(
            img_as_float(patch), nbins=self._bins)
        return histogram

    def distance(self, desc1, desc2):
        result = cosine(desc1, desc2)

        if self._thres:
            result = 0 if result < self._thres else 1

        return result

    def distance_braycurtis(self, desc1, desc2):
        return braycurtis(desc1, desc2)
