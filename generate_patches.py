import numpy as np
import warnings

from os import path, mkdir
from shutil import rmtree

from skimage.io import imread, imsave
from skimage.transform import ProjectiveTransform

from tools import get_keypoints_random
from image import get_patch


def read_image_set(base_path, extension='ppm'):
    img_set = []

    # Read first image
    img1 = imread('{}/img1.{}'.format(base_path, extension), as_grey=True)
    img_set.append((img1, None))

    # Read remaining images with corresponding matrices
    for index in range(2, 7):
        img = imread(path.join(base_path, 'img{}.{}'.format(
            index, extension)), as_grey=True)
        matrix = np.loadtxt(path.join(base_path, 'H1to{}p'.format(index)))
        img_set.append((img, matrix))

    return np.array(img_set)


def fits_in_image(image, keypoint, padding=32):
    kp_x, kp_y = keypoint
    img_h, img_w = image.shape

    return (kp_x >= padding and kp_x <= img_w - padding) and \
           (kp_y >= padding and kp_y <= img_h - padding)


def get_patch_name(image_index, kp_index):
    return 'img{}_patch{}'.format(image_index, kp_index)


def save_labels(labels, target_path):
    with open(path.join(target_path, 'labels.txt'), 'w') as labels_fp:
        for patch_name, img1_patch_name in labels.items():
            labels_fp.write('{} {}\n'.format(patch_name, img1_patch_name))


def read_labels(labels_path):
    labels = {}
    with open(labels_path, 'r') as labels_fp:
        for line in labels_fp.readline():
            print(line)


def generate_patches(img_set, target_path):
    if path.exists(target_path):
        rmtree(target_path)
    mkdir(target_path)

    img1 = img_set[0, 0]
    img_tuples = img_set[1:]

    img1_keypoints = get_keypoints_random(img1)
    img1_patches = [(get_patch_name(1, index), get_patch(img1, kp))
                    for index, kp in enumerate(img1_keypoints)]

    for patch_name, patch in img1_patches:
        patch_path = path.join(target_path, '{}.png'.format(patch_name))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(patch_path, patch)

    labels = {}

    for img_index, (img, matrix) in enumerate(img_tuples):
        trans = ProjectiveTransform(matrix)
        keypoints = trans(img1_keypoints)

        for kp_index, kp in enumerate(keypoints):
            if not fits_in_image(img, kp):
                continue

            img1_patch_name, img1_patch = img1_patches[kp_index]

            patch = get_patch(img, kp)
            # Remaining images index starts from 2
            patch_name = get_patch_name(img_index + 2, kp_index)

            # Save label in dictionary
            labels[patch_name] = img1_patch_name

            # Store image
            patch_path = path.join(target_path, '{}.png'.format(patch_name))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(patch_path, patch)

    save_labels(labels, target_path)


def main():
    names = [('blur-1', 'ppm'),
             ('blur-2', 'ppm'),
             ('jpeg-compression', 'ppm'),
             ('light', 'ppm'),
             ('viewpoint-1', 'ppm'),
             ('viewpoint-2', 'ppm'),
             ('zoom-rotation-1', 'ppm'),
             ('zoom-rotation-2', 'pgm')]

    for name, extension in names:
        try:
            print('Generating "{}" patches...'.format(name), end=' ')
            img_set = read_image_set(
                path.join('./images', name), extension=extension)
            generate_patches(img_set, path.join('./patches', name))
            print('Done.')
        except Exception as exception:
            print('Error generating patches from "{}" set'.format(name))
            print(exception)
            exit(1)


if __name__ == '__main__':
    main()
