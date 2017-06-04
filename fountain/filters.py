from fountain.data import *
import struct
from array import array
import numpy as np
import sh
import sys
import os
import glob

FILTERS_PER_BLOCK = 128

class Filters(Dataset):
    def __init__(self, dataset_name, num_filters, filter_width, angles, num_blocks=100, start_block=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.num_filters = num_filters
        self.filter_width = filter_width
        self.angles = angles
        infix = '{}_{}_{}_{}'.format(dataset_name, num_filters, filter_width, angles)
        self.infix = infix
        self.num_blocks = num_blocks
        self.start_block = start_block

    def build(self, src, dst):
        fns = glob.glob('{}/**/logs/filters_{}_*.tfrecords'.format(src, self.infix))
        sh.zip('-j', os.path.join(dst, 'filters_{}.zip'.format(self.infix)), *fns)

    def get_size(self):
        return self.num_filters * self.num_blocks

    def get_example_dtype(self):
        return tf.float32,

    def get_channels(self):
        if self.dataset_name.startswith('mnist'):
            return 1
        else:
            return 3

    def get_example_shape(self):
        return (self.filter_width, self.filter_width, self.get_channels())

    def files(self):
        with sub_path(self.get_sub_path()):
            with sub_path(self.infix):
                ofn = 'filters_{}.zip'.format(self.infix)
                zf = OnlineFile(ofn, 'http://cake.da.inf.ethz.ch:8080/' + ofn)
                fns = ['filters_{}_{}.tfrecords'.format(self.infix, i) for i in range(self.start_block, self.start_block + self.num_blocks)]
                tfrs = [ZippedFile(fn, zf, True) for fn in fns]
                return tfrs

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'filter_raw': tf.FixedLenFeature([], tf.string),
                })
        filter = tf.decode_raw(features['filter_raw'], tf.float32)
        eshape = self.get_example_shape()
        filter.set_shape(np.prod(eshape))
        filter = tf.reshape(filter, eshape)
        return filter,

if __name__ == '__main__':
    if len(sys.argv) > 1:
        src, dst = sys.argv[1:]
        ds, nf, fw, an = glob.glob('{}/**/logs/filters_*.tfrecords'.format(src))[0].rsplit('.', 1)[0].split('_')[1:-1]
        Filters(ds, int(nf), int(fw), int(an)).build(src, dst)
    else:
        print(Filters('cifar10', 128, 7, 8).create_queue())
