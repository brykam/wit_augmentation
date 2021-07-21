import numpy as np
from multiprocessing import Process, Queue, cpu_count
import cv2
import time
import noise


def generate_perlin_noise(image):
    shape = image.shape
    height, width, channels = shape
    scale = (height+width)/14
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    seed = np.random.randint(0, 100)

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
                                        # base=0
                                        )
    img = np.floor((world + 1) * 255 / 2).astype(np.uint8)
    return img


def add_perlin_noise(image, mask):
    weight = .07
    perlin_noise = generate_perlin_noise(image)
    img_and_perlin = cv2.addWeighted(
        image, 1. - weight, perlin_noise, weight, 0)
    segmented_img_perlin = np.zeros(image.shape,  dtype=np.uint8)
    for j, y in enumerate(mask):
        for i, color in enumerate(y):
            if color[0] == 0 and color[1] == 0 and color[2] == 255:
                segmented_img_perlin[j][i] = img_and_perlin[j][i]
            else:
                segmented_img_perlin[j][i] = image[j][i]
    return segmented_img_perlin


def process_data(thd, q, results):
    while True:
        data = q.get()
        if data == 'DONE':
            break
        # print(f"{thd.name} is processing image {data[0]}")
        result = add_perlin_noise(data[1], data[2])
        results.put([data[0], result])


class NoiseAdder(Process):
    def __init__(self, thread_id, name, q, results):
        Process.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.q = q
        self.results = results
        self.working = True

    def run(self):
        process_data(self, self.q, self.results)


class PerlinGenerator:
    def __init__(self, X, y, masks, no_cpus=None):
        if no_cpus == None or no_cpus > cpu_count():
            self.no_cpus = cpu_count() - 1
        elif no_cpus < 0:
            self.no_cpus = 1
        self.work_queue = Queue(self.no_cpus)
        self.processes = []
        self.process_id = 1
        self.masks = masks
        self.images = X
        self.labels = y
        self.results = Queue(len(self.images))
        self.augs = []
        self.mixed_labels = []
        self.augment_set()

    def create_processes(self):
        for _ in range(self.no_cpus):
            proc = NoiseAdder(
                self.process_id, f"Process-{self.process_id}", self.work_queue, self.results)
            proc.start()
            self.processes.append(proc)
            self.process_id += 1

    def queue_fill(self):
        for i, image in enumerate(self.images):
            self.work_queue.put([self.labels[i], image, self.masks[i]])
        for _ in range(len(self.processes)):
            self.work_queue.put('DONE')

    def augment_set(self):
        print('Starting perlin noise addition procedure')
        print(f'Using {self.no_cpus} processes')
        self.create_processes()
        self.queue_fill()

        for _ in range(len(self.images)):
            label, img = self.results.get()
            self.augs.append(img)
            self.mixed_labels.append(label)
        for proc in self.processes:
            proc.join()
        print('Finished adding noise')

    def get_augmented_set(self, amount=500):
        if amount > len(self.images):
            amount = len(self.images) - 1
        X = self.images + self.augs[:amount]
        y = self.labels + self.mixed_labels[:amount]
        return X, y


if __name__ == '__main__':
    from helpers import get_smear_set, get_mask_set, shuffle_dataset
    import time
    import numpy as np
    from sklearn.model_selection import KFold

    split_ratio = 0.8
    k_folds = 8
    X, y = get_smear_set()
    masks = get_mask_set()

    X_shuff, y_shuff, masks_shuff = shuffle_dataset(X, y, masks)

    t0 = time.time()
    pg = PerlinGenerator(X_shuff, y_shuff, masks_shuff)

    X, y = pg.get_augmented_set()
    print(len(X))
    print('time elapsed: ', time.time() - t0)
