import random
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import cv2
from perlin_generator import PerlinGenerator


def get_image_lists():
    walk_dir = os.path.join(os.getcwd(), 'pap-smear2005')
    file_list = list(glob.iglob(walk_dir + '**/**/*.BMP', recursive=True))

    normal_cells = []
    abnormal_cells = []

    for filename in file_list:
        if 'normal' in filename:
            normal_cells.append(filename)
        else:
            abnormal_cells.append(filename)
    return normal_cells, abnormal_cells


def load_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def shuffle_dataset(X, y, masks=None):
    if masks is None:
        dataset = list(zip(X, y))
        random.shuffle(dataset)
        shuffled_X, shuffled_y = zip(*dataset)
        return list(shuffled_X), list(shuffled_y)
    else:
        dataset = list(zip(X, y, masks))
        random.shuffle(dataset)
        shuffled_X, shuffled_y, shuffled_m = zip(*dataset)
        return list(shuffled_X), list(shuffled_y), list(shuffled_m)


def get_smear_set(with_labels=False):
    normal_cells, abnormal_cells = get_image_lists()
    images = []
    labels = []
    print("Healhy cell images: ", len(normal_cells))
    print("Unhealhy cell images: ", len(abnormal_cells))
    if not with_labels:
        for filename in normal_cells:
            images.append(load_image(filename))
            labels.append(0)

        for filename in abnormal_cells:
            images.append(load_image(filename))
            labels.append(1)
    else:
        for filename in normal_cells + abnormal_cells:
            images.append(load_image(filename))
            labels.append(filename.split('/')[-2])
    labels = np.array(labels).reshape(-1, 1)
    return images, labels  # , onehot_encoder.categories_


def augmentation_by_copy(X, y, amount=500):
    if amount > len(X):
        amount = len(X) - 1
    X_copy, y_copy = shuffle_dataset(X, y, None)

    X_aug = X + X_copy[:amount]
    y_aug = y + y_copy[:amount]
    return X_aug, y_aug


def augmentation_by_perlin(X, y, masks, amount=500):
    if amount > len(X):
        amount = len(X) - 1
    pg = PerlinGenerator(X, y, masks)
    X_aug, y_aug = pg.get_augmented_set(amount)
    return X_aug, y_aug


def get_mask_set():
    normal_file_cells, abnormal_cells = get_image_lists()
    images = []
    for filename in normal_file_cells:
        images.append(load_image(filename[:-4] + '-d.bmp'))
    for filename in abnormal_cells:
        images.append(load_image(filename[:-4] + '-d.bmp'))
    images = np.array(images)
    return images


if __name__ == '__main__':
    images, labels = get_smear_set()
    masks = get_mask_set()
    ax1 = plt.subplot(211)
    # ax1.imshow(add_perlin_noise(images[0], masks[0]))
    ax2 = plt.subplot(212)
    ax2.imshow(images[0])
    plt.show()

    X_full, y_full, categories = get_smear_set()
    X_aug, y_aug = augmentation_by_copy(X_full, y_full, 500)
    print("Size of augmented image set: ", len(X_aug))
    print("Size of augmented label set: ", len(y_aug))
