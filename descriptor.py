import numpy as np
from collections import namedtuple

from HuMomentsDescriptor import HuMomentsDescriptor
from ExposureDescriptor import ExposureDescriptor


Descriptor = namedtuple('Descriptor', ['moments', 'exposure'])

def trim_patch(patch, size=(20, 20)):
    img_h, img_w = patch.shape
    h, w = size
    diff_h, diff_w = img_h - h, img_w - w
    return patch.copy()[int(diff_h/2):-int(diff_h/2),
                        int(diff_w/2):-int(diff_w/2)]


def extract(image, keypoints):
    hu_moments = HuMomentsDescriptor(patch_size=(20, 20))
    exposure = ExposureDescriptor(patch_size=(32, 32))

    moments_descriptors = hu_moments.extract(image, keypoints)
    exposure_descriptors = exposure.extract(image, keypoints)
    assert(len(moments_descriptors) == len(exposure_descriptors))

    # Put it into Descriptor struct
    descriptors = [Descriptor(m, e) for m, e in zip(
        moments_descriptors, exposure_descriptors)]

    return descriptors


def extract_for_patch(patch):
    hu_moments = HuMomentsDescriptor()
    exposure = ExposureDescriptor()

    moments_desc = hu_moments.extract_for_patch(trim_patch(patch, size=(20, 20)))
    exposure_desc = exposure.extract_for_patch(patch)

    return Descriptor(moments_desc, exposure_desc)


def distance(desc1, desc2):
    return distance_weighted(desc1, desc2)


def distance_weighted(desc1, desc2, w_moments=0.4, w_exposure=0.6):
    assert(w_moments + w_exposure == 1)

    hu_moments = HuMomentsDescriptor()
    exposure = ExposureDescriptor()

    hu_moments_distance = hu_moments.distance_braycurtis(
        desc1.moments, desc2.moments)
    exposure_distance = exposure.distance(
        desc1.exposure, desc2.exposure)

    return (w_exposure * exposure_distance) + \
           (w_moments * hu_moments_distance)
