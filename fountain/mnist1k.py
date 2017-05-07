from fountain.data import *
import struct
from array import array
import numpy as np


BLOCK_SIZE = 1000
TOTAL_IMAGES_PER_DIGIT = 5000
TOTAL_BLOCKS_PER_DIGIT = math.ceil(TOTAL_IMAGES_PER_DIGIT / BLOCK_SIZE)
DIGIT_WEIGHTS = [1, 1, 10, 10, 10, 100, 100, 100, 1000, 1000]
DIGIT_WEIGHTS = np.array(DIGIT_WEIGHTS) / np.sum(DIGIT_WEIGHTS)

def get_img_shape(num_digits):
    return [28, 28 * num_digits, 1]

class MNIST1K(LabeledImageMixin, Dataset):
    def __init__(self, num_digits=2, num_blocks=10, start_block=0):
        super().__init__()
        self.num_digits = num_digits
        self.num_blocks = num_blocks
        self.start_block = start_block

    def get_size(self):
        return self.num_blocks * BLOCK_SIZE

    def files(self):
        with sub_path(self.get_sub_path()):
            filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
            tars = [OnlineFile(fn, 'http://yann.lecun.com/exdb/mnist/' + fn) for fn in filenames]
            ubytefiles = [GzippedFile(f.name[:-3], f) for f in tars]
            files = [self.MNIST1KDataFile('mnist1k_{}_{:04d}.tfrecords'.format(self.num_digits, b), ubytefiles, self.num_digits, b) for b in range(self.start_block, self.start_block + self.num_blocks)]
            return files

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        img_shape = get_img_shape(self.num_digits)
        image.set_shape(np.prod(img_shape))
        image = tf.reshape(image, img_shape)
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        label = tf.cast(features['label'], tf.int32)
        return image, label

    class MNIST1KDataFile(File):
        def __init__(self, name, ubyteFiles, num_digits, block):
            super().__init__(name, ubyteFiles)
            self.num_digits = num_digits
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
            pure_data = pure_data.reshape((-1, 28, 28, 1))
            pure_labels = np.concatenate((self.load_labels(self.dependencies[1].path), self.load_labels(self.dependencies[3].path)))
            assert len(pure_data) == len(pure_labels)

            pdbc = [pure_data[pure_labels == c] for c in range(10)]
            data, labels = [], []
            for _ in range(BLOCK_SIZE):
                lbls = self.rng.choice(10, self.num_digits, True, DIGIT_WEIGHTS)
                pd = [pdbc[l][self.rng.choice(len(pdbc[l]))] for l in lbls]
                data.append(np.hstack(pd))
                labels.append((10 ** np.arange(self.num_digits, dtype=np.int64)).dot(lbls))

            with tf.python_io.TFRecordWriter(self.path) as writer:
                for d, l in zip(data, labels):
                    image_raw = d.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l])),
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        }))
                    writer.write(example.SerializeToString())

if __name__ == '__main__':
    print(MNIST1K(num_digits=2, num_blocks=400).create_queue())
    print(MNIST1K(num_digits=3, num_blocks=400).create_queue())
