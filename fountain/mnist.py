from data import *
import struct
from array import array
import numpy as np


class MNIST(Dataset):
    def files(self):
        with sub_path('mnist'):
            filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
            tars = [OnlineFile(fn, 'http://yann.lecun.com/exdb/mnist/' + fn) for fn in filenames]
            ubytefiles = [GzippedFile(f.name[:-3], f) for f in tars]
            files = [self.MNISTDataFile(ubf.name + '.npy', 'labels' in ubf.name, ubf) for ubf in ubytefiles]
            return files

    def get_data_raw(self):
        files = self.files()
        images = np.concatenate((np.load(files[0].path), np.load(files[2].path)))
        labels = np.concatenate((np.load(files[1].path), np.load(files[3].path)))
        return images, labels

    class MNISTDataFile(File):
        def __init__(self, name, isLabels, ubyteFile):
            super().__init__(name, [ubyteFile])
            self.isLabels = isLabels
            self.ubyteFile = ubyteFile

        # https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
        def load_labels(self, path_lbl):
            with open(path_lbl, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                if magic != 2049:
                    raise ValueError('Magic number mismatch, expected 2049,'
                                     'got {}'.format(magic))

                labels = array("B", f.read())
                return labels

        def load_images(self, path_img):
            with open(path_img, 'rb') as f:
                magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051,'
                                     'got {}'.format(magic))

                image_data = array("B", f.read())

            images = []
            for i in range(size):
                images.append([0] * rows * cols)

            for i in range(size):
                images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

            return (np.array(images).reshape((size, 1, rows, cols)) / 255).astype(np.float32)

        def update(self):
            if self.isLabels:
                data = self.load_labels(self.ubyteFile.path)
            else:
                data = self.load_images(self.ubyteFile.path)
            np.save(self.path, data)


if __name__ == '__main__':
    print(MNIST().get_data()[0].shape)
