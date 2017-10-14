#!/usr/bin/env python3

from fountain.data import *
import numpy as np
from scipy.io import loadmat


IMG_SHAPE = [32, 32, 3]

class SVHN2(LabeledImageMixin, Dataset):
    def __init__(self, tfrecord=True, mode='train', dequant=True):
        super().__init__()
        self.mode = mode
        self.isTf = tfrecord
        self.dequant = dequant

    def get_size(self):
        return 73257 if self.mode == 'train' else 26032

    def files(self):
        with sub_path('svhn2'):
            mats = [OnlineFile('svhn_train.mat', 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'), OnlineFile('svhn_test.mat', 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat')]
            if not self.isTf:
                files = [self.SVHN2DataFile('svhn_images.npy', False, mats), self.SVHN2DataFile('svhn_labels.npy', True, mats)]
            else:
                if self.mode == 'train':
                    files = [self.SVHN2DataFile('svhn.train.tfrecords', None, [mats[0]])]
                else:
                    files = [self.SVHN2DataFile('svhn.test.tfrecords', None, [mats[1]])]
            return files

    def get_data_raw(self):
        files = self.files()
        images = np.load(files[0].path)
        labels = np.load(files[1].path)
        return images, labels

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape(np.prod(IMG_SHAPE))
        image = tf.reshape(image, IMG_SHAPE)
        image = tf.cast(image, tf.float32)
        if self.dequant:
            image = image + tf.random_uniform(image.get_shape(), 0., 1.)
            imge = image * (2. / 256) - 1.
        else:
            imge = image * (2. / 255) - 1.
        label = tf.cast(features['label'], tf.int32)
        return image, label

    class SVHN2DataFile(File):
        def __init__(self, name, isLabels, matfiles):
            super().__init__(name, matfiles)
            self.isLabels = isLabels
            self.isTf = name.endswith('tfrecords')

        def update(self):
            if not self.isTf:
                if self.isLabels:
                    key = 'y'
                    transp = (0, 1)
                else:
                    key = 'X'
                    transp = (3, 1, 0, 2)
                data = np.concatenate([loadmat(f.path)[key].transpose(transp) for f in self.dependencies if f.name.endswith('.mat')])

                if self.isLabels:
                    data = np.ravel(data.astype(np.int64)) - 1
                else:
                    data = data.astype(np.float32) / 255

                np.save(self.path, data)
            else:
                matds = loadmat(self.dependencies[0].path)
                data, labels = matds['X'].transpose(3, 1, 0, 2).astype(np.uint8), matds['y'].transpose(0, 1).ravel().astype(np.int64) - 1
                with tf.python_io.TFRecordWriter(self.path) as writer:
                    for d, l in zip(data, labels):
                        image_raw = d.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l])),
                            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                            }))
                        writer.write(example.SerializeToString())


if __name__ == '__main__':
    print(SVHN2(False).get_data()[0][:100])
    print(SVHN2().create_queue())
