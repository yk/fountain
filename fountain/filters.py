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
        dfmax = max([int(df.strip().rsplit('_', 1)[1].split('.')[0]) for df in glob.glob('{}/filters_{}_*.tfrecords'.format(dst, self.infix))] + [-1])
        for idx, fn in enumerate(glob.glob('{}/**/logs/filters.tfrecords'.format(src))):
            nfn = os.path.join(dst, 'filters_{}_{}.tfrecords'.format(self.infix, str(dfmax + idx + 1)))
            sh.cp(fn, nfn)
        os.chdir(dst)
        sh.zip('filters_{}.zip'.format(self.infix), *glob.glob('filters_{}_*.tfrecords'.format(self.infix)))

    def get_size(self):
        return self.num_filters * self.num_blocks

    def get_example_dtype(self):
        return tf.float32

    def get_example_shape(self):
        return (self.filter_width, self.filter_width, 1)

    def files(self):
        with sub_path(self.get_sub_path()):
            ofn = 'filters_{}.zip'.format(self.infix)
            zf = OnlineFile(ofn, 'http://cake.da.inf.ethz.ch:8080/' + ofn)
            fns = ['filters_{}_{}.tfrecords'.format(self.infix, i) for i in range(self.start_block, self.num_blocks)]
            tfrs = [ZippedFile(fn, zf, True) for fn in fns]
            return tfrs

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'filters_raw': tf.FixedLenFeature([], tf.string),
                })
        filters = tf.decode_raw(features['filters_raw'], tf.float32)
        eshape = self.get_example_shape()
        filters.set_shape(np.prod(eshape))
        filters = tf.reshape(filters, eshape)
        return filters

if __name__ == '__main__':
    if len(sys.argv) > 1:
        src, dst, ds, nf, fw, an = sys.argv[1:]
        Filters(ds, int(nf), int(fw), int(an)).build(src, dst)
    else:
        print(Filters('cifar10', 128, 7, 8).create_queue())
