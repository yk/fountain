#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle


IMG_SHAPE = [32, 32, 3]

class CIFAR10(LabeledImageMixin, Dataset):
    def __init__(self, tfrecord=True, mode='train', num_classes=10, dequant=True):
        super().__init__()
        self.mode = mode
        self.isTf = tfrecord
        assert num_classes == 10 or num_classes == 100
        self.num_classes = num_classes
        self.dequant = dequant

    def name(self):
        return 'cifar{}'.format(self.num_classes)

    def get_size(self):
        return 50000 if self.mode == 'train' else 10000

    def files(self):
        with sub_path(self.get_sub_path()):
            tar = GzippedFile('cifar{}.tar'.format(self.num_classes), OnlineFile('cifar{}.tar.gz'.format(self.num_classes), 'https://www.cs.toronto.edu/~kriz/cifar-{}-python.tar.gz'.format(self.num_classes)))
            if self.num_classes == 10:
                batches = [TaredFile('cifar{}_b{}.pkl'.format(self.num_classes, i), 'cifar-{}-batches-py/data_batch_{}'.format(self.num_classes, i), tar) for i in range(1, 6)] + [TaredFile('cifar{}_test'.format(self.num_classes), 'cifar-{}-batches-py/test_batch'.format(self.num_classes), tar)]
            else:
                batches = [TaredFile('cifar100_train.pkl', 'cifar-100-python/train', tar), TaredFile('cifar100_test.pkl', 'cifar-100-python/test', tar)]
            if not self.isTf:
                files = [self.CIFAR10DataFile('cifar{}_images.npy'.format(self.num_classes), self.num_classes, False, batches), self.CIFAR10DataFile('cifar{}_labels.npy'.format(self.num_classes), self.num_classes, True, batches)]
            else:
                if self.mode == 'train':
                    files = [self.CIFAR10DataFile('cifar{}.train.tfrecords'.format(self.num_classes), self.num_classes, None, batches[:-1])]
                else:
                    files = [self.CIFAR10DataFile('cifar{}.test.tfrecords'.format(self.num_classes), self.num_classes, None, batches[-1:])]
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
        if self.dequant:
            image = image + tf.random_uniform(image.get_shape(), 0., 1.)
            imge = image * (2. / 256) - 1.
        else:
            imge = image * (2. / 255) - 1.
        label = tf.cast(features['label'], tf.int32)
        return image, label

    class CIFAR10DataFile(File):
        def __init__(self, name, num_classes, isLabels, batches):
            super().__init__(name, batches)
            self.isLabels = isLabels
            self.isTf = name.endswith('tfrecords')
            self.num_classes = num_classes

        def update(self):
            labels_name = 'labels' if self.num_classes == 10 else 'fine_labels'
            if not self.isTf:
                data = []
                for b in self.dependencies:
                    with open(b.path, 'rb') as f:
                        d = cPickle.load(f, encoding='latin1')
                    if self.isLabels:
                        data.extend(d[labels_name])
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
                        ds, ls = np.array(ex['data'], np.uint8), np.array(ex[labels_name], np.int64)
                        ds = ds.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((-1, 32 * 32 * 3))
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
    print(CIFAR10(num_classes=100).create_queue())
