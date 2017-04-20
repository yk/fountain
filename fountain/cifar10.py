#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle


IMG_SHAPE = [32, 32, 3]

class CIFAR10(Dataset):
    def __init__(self, tfrecord=True, mode='train'):
        super().__init__(mode=mode)
        self.isTf = tfrecord

    def get_size(self):
        return 50000 if self.mode == 'train' else 10000

    def files(self):
        with sub_path('cifar10'):
            tar = GzippedFile('cifar10.tar', OnlineFile('cifar10.tar.gz', 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'))
            batches = [TaredFile('cifar10_b{}.pkl'.format(i), 'cifar-10-batches-py/data_batch_{}'.format(i), tar) for i in range(1, 6)] + [TaredFile('cifar10_test', 'cifar-10-batches-py/test_batch', tar)]
            if not self.isTf:
                files = [self.CIFAR10DataFile('cifar10_images.npy', False, batches), self.CIFAR10DataFile('cifar10_labels.npy', True, batches)]
            else:
                if self.mode == 'train':
                    files = [self.CIFAR10DataFile('cifar10.train.tfrecords', None, batches[:-1])]
                else:
                    files = [self.CIFAR10DataFile('cifar10.test.tfrecords', None, batches[-1:])]
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
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        label = tf.cast(features['label'], tf.int32)
        return image, label

    class CIFAR10DataFile(File):
        def __init__(self, name, isLabels, batches):
            super().__init__(name, batches)
            self.isLabels = isLabels
            self.isTf = name.endswith('tfrecords')

        def update(self):
            if not self.isTf:
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
            else:
                with tf.python_io.TFRecordWriter(self.path) as writer:
                    for b in self.dependencies:
                        with open(b.path, 'rb') as f:
                            ex = cPickle.load(f, encoding='latin1')
                        ds, ls = np.array(ex['data'], np.uint8), np.array(ex['labels'], np.int64)
                        for d, l in zip(ds, ls):
                            image_raw = d.tostring()
                            example = tf.train.Example(features=tf.train.Features(feature={
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l])),
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                                }))
                            writer.write(example.SerializeToString())


if __name__ == '__main__':
    print(CIFAR10(tfrecord=False).get_data()[0].shape)
    print(CIFAR10().create_queue())
