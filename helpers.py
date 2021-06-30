import random
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import noise
import cv2


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

def shuffle_dataset(X, y):
    dataset = list(zip(X, y))
    random.shuffle(dataset)
    shuffled_X, shuffled_y = zip(*dataset)
    return list(shuffled_X), shuffled_y

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
    return images, labels #, onehot_encoder.categories_

def augmentation_by_copy(X, y, amount=500):
    if amount > len(X):
        amount = len(X) - 1
    X_copy, y_copy = np.copy(X), np.copy(y)
    X_copy, y_copy = shuffle_dataset(X_copy, y_copy)
    X_copy = np.asarray(X_copy)
    y_copy = np.asarray(y_copy)

    X_aug = np.concatenate((X, X_copy[:amount]), axis=0)
    y_aug = np.concatenate((y, y_copy[:amount]), axis=0)
    return X_aug, y_aug

def augmentation_by_perlin(X, y, amount=500):
    if amount > len(X):
        amount = len(X) - 1
    images, labels = get_smear_set()
    masks = get_mask_set()
    X_perlin = []
    y_perlin = []
    for i, image in enumerate(images):
        if i % 10 == 0:
            print(f"Adding Perlin to: {i} image")
        X_perlin.append(add_perlin_noise(image, masks[i]))
        y_perlin.append(labels[i])
    X_perlin, y_perlin = shuffle_dataset(X_perlin, y_perlin)
    X_perlin = np.asarray(X_perlin)
    y_perlin = np.asarray(y_perlin)
    X_aug = np.concatenate((X, X_perlin[:amount]), axis=0)
    y_aug = np.concatenate((y, y_perlin[:amount]), axis=0)
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

def generate_perlin_noise(image):
    shape = image.shape
    height, width, channels = shape
    scale = (height+width)/14
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    seed = np.random.randint(0,100)

    world = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        #base=0
                                        )
    img = np.floor((world + 1) * 255 /2).astype(np.uint8)
    return img


def add_perlin_noise(image, mask):
    weight = .07
    perlin_noise = generate_perlin_noise(image)
    img_and_perlin = cv2.addWeighted(image, 1. - weight, perlin_noise, weight, 0)
    segmented_img_perlin = np.zeros(image.shape,  dtype = np.uint8)
    for j, y in enumerate(mask):
        for i, color in enumerate(y):
            if color[0] == 0 and color[1] == 0 and color[2] !=0:
                segmented_img_perlin[j][i] = img_and_perlin[j][i]
            else:
                segmented_img_perlin[j][i] = image[j][i]
    return segmented_img_perlin

if __name__ == '__main__':
    images, labels = get_smear_set()
    masks = get_mask_set()
    ax1 = plt.subplot(211)
    ax1.imshow(add_perlin_noise(images[0], masks[0]))
    ax2 = plt.subplot(212)
    ax2.imshow(images[0])
    plt.show()

    # X_full, y_full, categories = get_smear_set()
    # X_aug, y_aug = augmentation_by_copy(X_full, y_full, 500)
    # print("Size of augmented image set: ", len(X_aug))
    # print("Size of augmented label set: ", len(y_aug))