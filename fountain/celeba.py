#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import itertools as itt
from fountain.utils import jpg2npy, to_chunks, get_chunk
import functools as fct
import math
import tensorflow as tf


BLOCK_SIZE = 1000
TOTAL_IMAGES = 202599
TOTAL_BLOCKS = math.ceil(TOTAL_IMAGES / BLOCK_SIZE)
NUM_LABELS = 40
IMG_SHAPE = [218, 178, 3]
GOOD_LABELS = [0, 2, 5]

class CelebA(LabeledImageMixin, Dataset):
    def __init__(self, num_blocks=10, start_block=0, resize=None):
        super().__init__()
        self.num_blocks = num_blocks
        self.start_block = start_block
        self.num_images = num_blocks * BLOCK_SIZE
        self.start_image = start_block * BLOCK_SIZE
        self.resize = resize

    def get_size(self):
        return self.num_blocks * BLOCK_SIZE

    def files(self):
        with sub_path(self.get_sub_path()):
            jpgs = [ZippedFile('img_align_celeba/{}.jpg'.format(str(i).zfill(6)), OnlineFile('img_align_celeba.zip', 'http://cake.da.inf.ethz.ch:8080/img_align_celeba.zip'), extract_all=True) for i in range(self.start_image + 1, self.start_image + self.num_images + 1)]
            lbls = OnlineFile('list_attr_celeba.txt', 'http://cake.da.inf.ethz.ch:8080/list_attr_celeba.txt')
            files = [self.CelebADataFile('celeba_{}{:03d}.tfrecords'.format('{}x{}_'.format(*self.resize) if self.resize else '', b + self.start_block), j, lbls, b + self.start_block, resize) for b, j in enumerate(to_chunks(jpgs, BLOCK_SIZE))]
            return files

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'labels': tf.FixedLenFeature([NUM_LABELS], tf.int64),
                })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        img_shape = IMG_SHAPE[:]
        if self.resize:
            img_shape[:2] = self.resize
        image.set_shape(np.prod(img_shape))
        image = tf.reshape(image, img_shape)
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        if self.resize:
            image = tf.image.resize_images(image, self.resize)
        labels = tf.cast(features['labels'], tf.int32)
        labels = labels[0] * 1 + labels[2] * 2 + labels[5] * 4
        return image, labels

    class CelebADataFile(File):
        def __init__(self, name, deps, labelsFile, block, resize=None):
            super().__init__(name, deps + [labelsFile])
            self.labelsFile = labelsFile
            self.block = block
            self.resize = resize

        def update(self):
            with ThreadPoolExecutor() as ex:
                mp = ex.map(fct.partial(jpg2npy, resize=self.resize), [b.path for b in self.dependencies[:-1]])
                data = list(mp)
            assert np.max(data) == 255
            assert np.min(data) == 0

            with open(self.labelsFile.path) as f:
                lines = [l.strip().split()[1:] for l in list(f.readlines())[2:]]
                lines = get_chunk(lines, BLOCK_SIZE, self.block)
            labels = (np.array(lines, dtype=np.int64) + 1) // 2
            assert np.max(labels) == 1
            assert np.min(labels) == 0
            assert labels.shape[1] == NUM_LABELS, str(len(labels))

            assert len(data) == len(labels)
            with tf.python_io.TFRecordWriter(self.path) as writer:
                for d, l in zip(data, labels):
                    image_raw = d.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=l)),
                    }))
                    writer.write(example.SerializeToString())

if __name__ == '__main__':
    with tf.Graph().as_default():
        print(CelebA().create_queue())
