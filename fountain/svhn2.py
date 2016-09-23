#!/usr/bin/env python3

from fountain.data import *
import numpy as np
from scipy.io import loadmat


class SVHN2(Dataset):
    def files(self):
        with sub_path('svhn2'):
            mats = [OnlineFile('svhn_train.mat', 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'), OnlineFile('svhn_test.mat', 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat')]
            files = [self.SVHN2DataFile('svhn_images.npy', False, mats), self.SVHN2DataFile('svhn_labels.npy', True, mats)]
            return files

    def get_data_raw(self):
        files = self.files()
        images = np.load(files[0].path)
        labels = np.load(files[1].path)
        return images, labels

    class SVHN2DataFile(File):
        def __init__(self, name, isLabels, matfiles):
            super().__init__(name, matfiles)
            self.isLabels = isLabels

        def update(self):
            if self.isLabels:
                key = 'y'
                transp = (0, 1)
            else:
                key = 'X'
                transp = (3, 1, 0, 2)
            data = np.concatenate([loadmat(f.path)[key].transpose(transp) for f in self.dependencies if f.name.endswith('.mat')])

            if self.isLabels:
                data = np.ravel(data)

            np.save(self.path, data)


if __name__ == '__main__':
    print(SVHN2().get_data()[1].shape)
