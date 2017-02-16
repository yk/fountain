#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import itertools as itt
from fountain.utils import jpg2npy


class CelebA(Dataset):
    def files(self):
        with sub_path('celeba'):
            jpgs = [ZippedFile('img_align_celeba/{}.jpg'.format(str(i).zfill(6)), OnlineFile('img_align_celeba.zip', 'http://cake.da.inf.ethz.ch:8080/img_align_celeba.zip'), extract_all=True) for i in range(1, 202600)]
            files = [self.CelebADataFile('celeba_images.npy', jpgs, False), self.CelebADataFile('celeba_labels.npy', [OnlineFile('list_attr_celeba.txt', 'http://cake.da.inf.ethz.ch:8080/list_attr_celeba.txt')], True)]
            return files

    def get_data_raw(self):
        files = self.files()
        images = np.load(files[0].path)
        labels = np.load(files[1].path)
        return images, labels

    class CelebADataFile(File):
        def __init__(self, name, deps, isLabels):
            super().__init__(name, deps)
            self.isLabels = isLabels

        def update(self):
            if self.isLabels:
                with open(self.dependencies[0].path) as f:
                    lines = [l.strip().split()[1:] for l in list(f.readlines())[2:]]
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
