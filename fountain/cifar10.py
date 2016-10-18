#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle


class CIFAR10(Dataset):
    def files(self):
        with sub_path('cifar10'):
            tar = GzippedFile('cifar10.tar', OnlineFile('cifar10.tar.gz', 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'))
            batches = [TaredFile('cifar10_b{}.pkl'.format(i), 'cifar-10-batches-py/data_batch_{}'.format(i), tar) for i in range(1, 6)] + [TaredFile('cifar10_test', 'cifar-10-batches-py/test_batch', tar)]
            files = [self.CIFAR10DataFile('cifar10_images.npy', False, batches), self.CIFAR10DataFile('cifar10_labels.npy', True, batches)]
            return files

    def get_data_raw(self):
        files = self.files()
        images = np.load(files[0].path)
        labels = np.load(files[1].path)
        return images, labels

    class CIFAR10DataFile(File):
        def __init__(self, name, isLabels, batches):
            super().__init__(name, batches)
            self.isLabels = isLabels

        def update(self):
            data = []
            for b in self.dependencies:
                with open(b.path, 'rb') as f:
                    d = cPickle.load(f, encoding='latin1')
                if self.isLabels:
                    data.extend(d['labels'])
                else:
                    data.extend(d['data'])
            if self.isLabels:
                data = np.array(data, dtype=np.int32)
            else:
                data = np.array(data, dtype=np.float32) / 255.
                data = data.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
            np.save(self.path, data)



if __name__ == '__main__':
    print(CIFAR10().get_data()[0].shape)
