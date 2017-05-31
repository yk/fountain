from fountain.data import *
import struct
from array import array
import numpy as np


IMG_SHAPE = [28, 28, 1]

class MNIST(LabeledImageMixin, Dataset):
    def __init__(self, tfrecord=True, mode='train'):
        super().__init__()
        self.mode = mode
        self.isTf = tfrecord

    def get_size(self):
        return 60000 if self.mode == 'train' else 10000

    def files(self):
        with sub_path('mnist'):
            filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
            tars = [OnlineFile(fn, 'http://yann.lecun.com/exdb/mnist/' + fn) for fn in filenames]
            ubytefiles = [GzippedFile(f.name[:-3], f) for f in tars]
            if not self.isTf:
                files = [self.MNISTDataFile(ubf.name + '.npy', 'labels' in ubf.name, ubf) for ubf in ubytefiles]
            else:
                if self.mode == 'train':
                    files = [self.MNISTDataFile('mnist.train.tfrecords', None, ubytefiles[:2])]
                else:
                    files = [self.MNISTDataFile('mnist.test.tfrecords', None, ubytefiles[2:])]
            return files

    def get_data_raw(self):
        files = self.files()
        images = np.concatenate((np.load(files[0].path), np.load(files[2].path)))
        labels = np.concatenate((np.load(files[1].path), np.load(files[3].path)))
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

    class MNISTDataFile(File):
        def __init__(self, name, isLabels, ubyteFile):
            self.isTf = name.endswith('tfrecords')
            super().__init__(name, ubyteFile if self.isTf else [ubyteFile])
            self.isLabels = isLabels
            self.ubyteFile = ubyteFile

        # https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
        def load_labels(self, path_lbl):
            with open(path_lbl, 'rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                if magic != 2049:
                    raise ValueError('Magic number mismatch, expected 2049,'
                                     'got {}'.format(magic))

                labels = array("B", f.read())
                return np.array(labels, dtype=np.int64) if self.isTf else labels

        def load_images(self, path_img):
            with open(path_img, 'rb') as f:
                magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051,'
                                     'got {}'.format(magic))

                image_data = array("B", f.read())

            images = []
            # for i in range(size):
                # images.append([0] * rows * cols)

            for i in range(size):
                images.append(np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype=np.uint8))

            if not self.isTf:
                images = (np.array(images) / 255).astype(np.float32)
            return images

        def update(self):
            if not self.isTf:
                if self.isLabels:
                    data = self.load_labels(self.ubyteFile.path)
                else:
                    data = self.load_images(self.ubyteFile.path)
                np.save(self.path, data)
            else:
                data = self.load_images(self.ubyteFile[0].path)
                labels = self.load_labels(self.ubyteFile[1].path)

                with tf.python_io.TFRecordWriter(self.path) as writer:
                    for d, l in zip(data, labels):
                        image_raw = d.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[l])),
                            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                            }))
                        writer.write(example.SerializeToString())

if __name__ == '__main__':
    print(np.max(MNIST(tfrecord=False).get_data()[0][:2]))
    print(MNIST().create_queue())
