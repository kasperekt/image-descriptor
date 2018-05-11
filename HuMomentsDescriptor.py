import numpy as np

from scipy import spatial
from skimage.measure import moments_hu
from image import get_patch

class HuMomentsDescriptor:
  def __init__(self, patch_size=(20, 20)):
    self._patch_size = patch_size

  def extract(self, image, keypoints):
    descriptors = [self.extract_for_keypoint(image, kp) for kp in keypoints]
    return np.array(descriptors)

  def extract_for_keypoint(self, image, keypoint):
    patch = get_patch(image, keypoint, size=self._patch_size)
    moments = moments_hu(patch)
    normed_moments = -np.sign(moments)*np.log10(np.abs(moments))

    return moments

  def distance(self, desc1, desc2):
    return spatial.distance.cosine(desc1, desc2)

  def distance_correlation(self, desc1, desc2):
    return spatial.distance.correlation(desc1, desc2)

  def distance_chebyshev(self, desc1, desc2):
    return spatial.distance.chebyshev(desc1, desc2)

  def distance_canberra(self, desc1, desc2):
    return spatial.distance.canberra(desc1, desc2)

  def distance_braycurtis(self, desc1, desc2):
    return spatial.distance.braycurtis(desc1, desc2)
