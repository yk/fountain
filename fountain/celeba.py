#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import itertools as itt
from fountain.utils import jpg2npy, to_chunks, get_chunk
import math
import tensorflow as tf


BLOCK_SIZE = 1000
TOTAL_IMAGES = 202599
TOTAL_BLOCKS = math.ceil(TOTAL_IMAGES / BLOCK_SIZE)
NUM_LABELS = 40
IMG_SHAPE = [218, 178, 3]

class CelebA(Dataset):
    def __init__(self, num_blocks=10, start_block=0):
        super().__init__()
        self.num_blocks = num_blocks
        self.start_block = start_block
        self.num_images = num_blocks * BLOCK_SIZE
        self.start_image = start_block * BLOCK_SIZE

    def files(self):
        with sub_path(self.get_sub_path()):
            jpgs = [ZippedFile('img_align_celeba/{}.jpg'.format(str(i).zfill(6)), OnlineFile('img_align_celeba.zip', 'http://cake.da.inf.ethz.ch:8080/img_align_celeba.zip'), extract_all=True) for i in range(self.start_image + 1, self.start_image + self.num_images + 1)]
            lbls = OnlineFile('list_attr_celeba.txt', 'http://cake.da.inf.ethz.ch:8080/list_attr_celeba.txt')
            files = [self.CelebADataFile('celeba_{:03d}.tfrecords'.format(b), j, lbls, b + self.start_block) for b, j in enumerate(to_chunks(jpgs, BLOCK_SIZE))]
            return files

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'labels': tf.FixedLenFeature([], tf.int64),
                })
        image = tf.decode_raw(features['image_raw'], tf.unit8)
        image = image.set_shape(IMG_SHAPE)
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        labels = tf.cast(features['labels'], tf.int32)
        return image, labels

    class CelebADataFile(File):
        def __init__(self, name, deps, labelsFile, block):
            super().__init__(name, deps)
            self.labelsFile = labelsFile
            self.block = block

        def update(self):
            with ThreadPoolExecutor() as ex:
                mp = ex.map(jpg2npy, [b.path for b in self.dependencies])
                data = list(mp)
            data = np.array(data).astype(np.uint8)
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
