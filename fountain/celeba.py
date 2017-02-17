#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import itertools as itt
from fountain.utils import jpg2npy, to_chunks, get_chunk
import math


BLOCK_SIZE = 1000
TOTAL_IMAGES = 202599
TOTAL_BLOCKS = math.ceil(TOTAL_IMAGES / BLOCK_SIZE)

class CelebA(Dataset):
    def __init__(self, num_blocks=10):
        super().__init__()
        self.num_blocks = num_blocks

    def files(self):
        with sub_path('celeba'):
            jpgs = [ZippedFile('img_align_celeba/{}.jpg'.format(str(i).zfill(6)), OnlineFile('img_align_celeba.zip', 'http://cake.da.inf.ethz.ch:8080/img_align_celeba.zip'), extract_all=True) for i in range(1, 202600)]
            lbls = OnlineFile('list_attr_celeba.txt', 'http://cake.da.inf.ethz.ch:8080/list_attr_celeba.txt')
            tuples = [(self.CelebADataFile('celeba_images_{}.npy'.format(b), j, False, b), self.CelebADataFile('celeba_labels_{}.npy'.format(b), [lbls], True, b)) for b, j in enumerate(to_chunks(jpgs, BLOCK_SIZE))]
            files = list(itt.chain.from_iterable(tuples))
            return files

    def get_data_raw(self):
        files = self.files()[:self.num_blocks * 2]
        npys = [np.load(f.path) for f in files]
        images = npys[0::2]
        labels = npys[1::2]
        images, labels = map(np.concatenate, (images, labels))
        labels = labels[:, [0, 2, 5]]
        labels *= [1, 2, 4]
        labels = np.sum(labels, axis=1)
        return images, labels

    class CelebADataFile(File):
        def __init__(self, name, deps, isLabels, block):
            super().__init__(name, deps)
            self.isLabels = isLabels
            self.block = block

        def update(self):
            if self.isLabels:
                with open(self.dependencies[0].path) as f:
                    lines = [l.strip().split()[1:] for l in list(f.readlines())[2:]]
                    lines = get_chunk(lines, BLOCK_SIZE, self.block)
                labels = (np.array(lines, dtype=np.int32) + 1) // 2
                assert np.max(labels) == 1
                assert np.min(labels) == 0
                np.save(self.path, labels)
            else:
                with ThreadPoolExecutor() as ex:
                    mp = ex.map(jpg2npy, [b.path for b in self.dependencies])
                    data = list(mp)
                data = np.array(data).astype(np.float32)
                assert np.max(data) == 255.
                assert np.min(data) == 0.
                np.save(self.path, data)


if __name__ == '__main__':
    print(CelebA().get_data()[0].shape)
