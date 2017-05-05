from fountain.data import *
import struct
from array import array
import numpy as np


IMG_SHAPE = [28, 28 * 3, 1]
BLOCK_SIZE = 1000
TOTAL_IMAGES = 5000000
TOTAL_BLOCKS = math.ceil(TOTAL_IMAGES / BLOCK_SIZE)

class MNIST1K(LabeledImageMixin, Dataset):
    def __init__(self, num_blocks=10, start_block=0):
        super().__init__()
        self.num_blocks = num_blocks
        self.start_block = start_block
        self.num_images = num_blocks * BLOCK_SIZE
        self.start_image = start_block * BLOCK_SIZE

    def get_size(self):
        return self.num_blocks * BLOCK_SIZE

    def files(self):
        with sub_path(self.get_sub_path()):
            filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
            tars = [OnlineFile(fn, 'http://yann.lecun.com/exdb/mnist/' + fn) for fn in filenames]
            ubytefiles = [GzippedFile(f.name[:-3], f) for f in tars]
            files = [self.MNIST1KDataFile('mnist1k_{:04d}.tfrecords'.format(b), ubytefiles, b) for b in range(self.start_block, self.start_block + self.num_blocks)]
            return files

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

    class MNIST1KDataFile(File):
        def __init__(self, name, ubyteFiles, block):
            super().__init__(name, ubyteFiles)
            self.block = block
            self.rng = np.random.RandomState(seed=11332244 + block)

        # https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
        def load_labels(self, path_lbl):
            with open(path_lbl, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                if magic != 2049:
                    raise ValueError('Magic number mismatch, expected 2049,'
                                     'got {}'.format(magic))

                labels = array("B", f.read())
                return np.array(labels, dtype=np.int64)

        def load_images(self, path_img):
            with open(path_img, 'rb') as f:
                magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051,'
                                     'got {}'.format(magic))

                image_data = array("B", f.read())

            images = []

            for i in range(size):
                images.append(np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype=np.uint8))

            return images

        def update(self):
            pure_data = np.concatenate((self.load_images(self.dependencies[0].path), self.load_images(self.dependencies[2].path)))
            pure_labels = np.concatenate((self.load_labels(self.dependencies[1].path), self.load_labels(self.dependencies[3].path)))
            assert len(data) == len(labels)
            data, labels = [], []
            for _ in range(BLOCK_SIZE):
                inds = self.rng.randint(0, len(data), 3)
                data.append(np.hstack(pure_data[inds]))
                labels.append(np.array([100, 10, 1], dtype=np.int64).dot(pure_labels[inds]))

            with tf.python_io.TFRecordWriter(self.path) as writer:
                for d, l in zip(data, labels):
                    image_raw = d.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l])),
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        }))
                    writer.write(example.SerializeToString())

if __name__ == '__main__':
    print(MNIST1K(num_blocks=TOTAL_IMAGES // BLOCK_SIZE).create_queue())
