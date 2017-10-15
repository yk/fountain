#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle


TOTAL_TRAIN = 1281167
TOTAL_TEST = 50000
BLOCK_SIZE = 1000

class ImageNet(LabeledImageMixin, Dataset):
    def __init__(self, mode='train', width=64, num_blocks=None, dequant=True):
        super().__init__()
        self.mode = mode
        self.width = width
        if num_blocks is None:
            num_blocks = TOTAL_TRAIN // BLOCK_SIZE if mode == 'train' else TOTAL_TEST // BLOCK_SIZE
        self.num_blocks = num_blocks
        self.dequant = dequant

    def get_example_shape(self):
        return [self.width, self.width, 3]

    def get_size(self):
        return self.num_blocks * BLOCK_SIZE

    def files(self):
        with sub_path(self.get_sub_path()):
            with sub_path('{}'.format(self.width)):
                if self.width == 64 and self.mode == 'train':
                    onl = [
                            OnlineFile('imagenet_train_1.zip', 'http://cake.da.inf.ethz.ch:8080/Imagenet64_train_part1.zip'),
                            OnlineFile('imagenet_train_2.zip', 'http://cake.da.inf.ethz.ch:8080/Imagenet64_train_part2.zip')
                            ]
                else:
                    onl = [OnlineFile('imagenet_{}.zip'.format(self.mode), 'http://cake.da.inf.ethz.ch:8080/Imagenet{}_{}.zip'.format(self.width, 'train' if self.mode == 'train' else 'val'))]
                if self.mode == 'train':
                    if self.width == 64:
                        batches = [ZippedFile('train_data_batch_{}'.format(b), onl[0], True) for b in range(1, 6)]\
                                + [ZippedFile('train_data_batch_{}'.format(b), onl[1], True) for b in range(6, 11)]
                    else:
                        batches = [ZippedFile('train_data_batch_{}'.format(b), onl[0], True) for b in range(1, 11)]
                else:
                    batches = [ZippedFile('val_data', onl[0], True)]
                files = [self.ImageNetDataFile('imagenet_{}_{}.tfrecords'.format(self.mode, b), self.width, self.mode, batches, b) for b in range(self.num_blocks)]
                return files

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        img_shape = self.get_example_shape()
        image.set_shape(np.prod(img_shape))
        image = tf.reshape(image, img_shape)
        image = tf.cast(image, tf.float32)
        if self.dequant:
            image = image + tf.random_uniform(image.get_shape(), 0., 1.)
            image = image * (2. / 256) - 1.
        else:
            image = image * (2. / 255) - 1.
        label = tf.cast(features['label'], tf.int32)
        return image, label

    class ImageNetDataFile(File):
        def __init__(self, name, width, mode, batches, block):
            super().__init__(name, batches)
            self.block = block
            self.width = width
            self.mode = mode

        def update(self):
            data = []
            labels = []
            for b in self.dependencies:
                with open(b.path, 'rb') as f:
                    ex = cPickle.load(f, encoding='latin1')
                    data.append(np.array(ex['data'], np.uint8))
                    labels.append(np.array(ex['labels'], np.int64))
            data = np.concatenate(data, 0)
            labels = np.concatenate(labels, 0)
            data = data.reshape((-1, 3, self.width, self.width)).transpose((0, 2, 3, 1)).reshape((-1, self.width * self.width * 3))
            total_blocks = TOTAL_TRAIN // BLOCK_SIZE if self.mode == 'train' else TOTAL_TEST // BLOCK_SIZE
            for b in range(total_blocks):
                with tf.python_io.TFRecordWriter(self.path.rsplit('_', 1)[0] + '_{}.tfrecords'.format(b)) as writer:
                    ind_start = b * BLOCK_SIZE
                    ind_stop = ind_start + BLOCK_SIZE
                    for d, l in zip(data[ind_start:ind_stop], labels[ind_start:ind_stop]):
                        image_raw = d.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l])),
                            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                            }))
                        writer.write(example.SerializeToString())


if __name__ == '__main__':
    print(ImageNet().create_queue())
