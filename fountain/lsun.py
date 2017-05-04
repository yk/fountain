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
import lmdb
import io


CATEGORIES = ["bedroom","bridge","church_outdoor","classroom","conference_room","dining_room","kitchen","living_room","restaurant","tower"]
BLOCK_SIZE = 1000
TOTAL_IMAGES = [3033042, 818687, 126227, 168103, 229069, 657571, 2212277, 1315802, 626331, 708264]
TOTAL_BLOCKS = [math.ceil(ti / BLOCK_SIZE) for ti in TOTAL_IMAGES]

def get_lsun_category_number(category):
    return CATEGORIES.index(category)

class LSUN(LabeledImageMixin, Dataset):
    def __init__(self, categories, mode='train', num_blocks=10, start_block=0, resize=None, self.crop=None):
        super().__init__()
        self.categories = categories
        self.num_blocks = num_blocks
        self.start_block = start_block
        self.num_images = num_blocks * BLOCK_SIZE
        self.start_image = start_block * BLOCK_SIZE
        if resize is None:
            resize = [64, 64]
        self.resize = resize
        self.crop = crop
        self.mode = mode

    def get_size(self):
        return self.num_blocks * BLOCK_SIZE * len(self.categories)

    def files(self):
        with sub_path(self.get_sub_path()):
            zips = [OnlineFile('{}_{}_lmdb.zip'.format(c, self.mode), 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag=latest&category={}&set={}'.format(c, self.mode)) for c in self.categories]
            lmdbs = [ZippedFile('{}/data.mdb'.format(z.name[:-4]), z, True) for z in zips]
            files = list(itt.chain.from_iterable([[self.LSUNDataFile('lsun_{}_{}_{}{}{:04d}.tfrecords'.format(c, self.mode, 'c{}x{}_'.format(*self.crop) if self.crop else '','{}x{}_'.format(*self.resize) if self.resize else '', b), db, c, b, self.resize, self.crop) for b in range(self.start_block, self.start_block + self.num_blocks)] for db, c in zip(lmdbs, self.categories)]))
            return files

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        img_shape = self.resize + [3]
        image.set_shape(np.prod(img_shape))
        image = tf.reshape(image, img_shape)
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        label = tf.cast(features['label'], tf.int32)
        return image, label

    class LSUNDataFile(File):
        def __init__(self, name, db, category, block, resize=None, crop=None):
            super().__init__(name, [db])
            self.category = category
            self.block = block
            self.resize = resize
            self.crop = crop

        def update(self):
            db = self.dependencies[0]
            env = lmdb.open('/'.join(db.path.split('/')[:-1]), map_size=1099511627776, max_readers=100, readonly=True)
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                mp = []
                for k, v in itt.islice(cursor, self.block * BLOCK_SIZE, (self.block + 1) * BLOCK_SIZE):
                    bio = io.BytesIO(v)
                    mp.append(jpg2npy(bio, resize=self.resize, crop=self.crop))
            assert np.max(data) == 255
            assert np.min(data) == 0

            labels = np.ones(len(data), dtype=np.int64) * np.int64(get_lsun_category_number(self.category))
            assert np.min(labels) >= 0
            assert np.max(labels) < len(CATEGORIES)

            assert len(data) == len(labels)
            with tf.python_io.TFRecordWriter(self.path) as writer:
                for d, l in zip(data, label):
                    image_raw = d.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=[l])),
                    }))
                    writer.write(example.SerializeToString())

if __name__ == '__main__':
    with tf.Graph().as_default():
        print(LSUN(['bedroom', 'kitchen', 'tower'], num_blocks=500, resize=[32, 32]).create_queue())
        print(LSUN(['bedroom', 'kitchen', 'tower'], num_blocks=500, resize=[64, 64]).create_queue())
