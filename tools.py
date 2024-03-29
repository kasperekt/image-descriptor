import numpy as np
from skimage.feature import corner_harris, corner_peaks


def invert_coords(coords):
    return np.array([[c[1], c[0]] for c in coords])


def trim_coords_to_shape(shape, coords, padding=32):
    height, width = shape

    def filter_fn(coord):
        x, y = coord
        return x >= padding and \
            x <= width - padding and \
            y >= padding and \
            y <= height - padding

    filtered = filter(filter_fn, coords)
    return np.array(list(filtered))


def get_keypoints(image):
    keypoints = corner_peaks(corner_harris(image), min_distance=5)
    keypoints = invert_coords(keypoints)
    keypoints = trim_coords_to_shape(image.shape, keypoints)
    return keypoints


def get_keypoints_random(image, size=150, padding=32):
    img_h, img_w = image.shape
    widths = np.random.uniform(padding, img_w - padding, size=(size, 1))
    heights = np.random.uniform(padding, img_h - padding, size=(size, 1))
    return np.hstack((widths, heights))
