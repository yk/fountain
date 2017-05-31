#!/usr/bin/env python3

import os.path
from fountain.utils import *
from contextlib import contextmanager
import logging
import time
import numpy as np
from filelock import FileLock
import tensorflow as tf
from rcfile import rcfile

rcf = rcfile('fountain')

BASE_DATA_PATH = rcf.get('base_data_path', os.path.expanduser("~/data"))
DATA_PATH = BASE_DATA_PATH


@contextmanager
def sub_path(path, relative=True):
    global DATA_PATH
    _old_dp = DATA_PATH
    DATA_PATH = os.path.join(_old_dp if relative else BASE_DATA_PATH, path)
    os.makedirs(DATA_PATH, mode=0o755, exist_ok=True)  # noqa
    yield
    DATA_PATH = _old_dp


# def wait_for_unlocked():
    # logging.debug('waiting for data path to be unlocked')
    # while os.path.exists(os.path.join(BASE_DATA_PATH, 'lock')):
        # time.sleep(1)


# @contextmanager
# def data_lock():
    # lockpath = os.path.join(BASE_DATA_PATH, 'lock')
    # with open(lockpath, 'w') as f:
        # f.write('locked')
    # yield
    # try:
        # os.remove(lockpath)
    # except:
        # pass

data_lock = FileLock(os.path.join(BASE_DATA_PATH, 'lock'))
logging.getLogger('filelock').setLevel(logging.WARNING)



class File:
    def __init__(self, name, dependencies=None):
        self.name = name
        self.path = os.path.join(DATA_PATH, self.name)
        self.dependencies = dependencies or []

    def exists(self):
        return os.path.exists(self.path)

    def last_modified(self):
        return os.path.getmtime(self.path)

    def update(self):
        raise Exception('Not Implemented')

    def ensure_updated(self, min_mtime=0.):
        dep_mtimes = [dep.ensure_updated(min_mtime) for dep in self.dependencies] + [0.]
        # wait_for_unlocked()
        if not self.exists() or self.last_modified() < max(min_mtime, max(dep_mtimes)): # check without lock to make faster
            with data_lock:
                if not self.exists() or self.last_modified() < max(min_mtime, max(dep_mtimes)):
                    print('updating {}'.format(self.name))
                    # with data_lock():
                    self.update()
        return self.last_modified()


class OnlineFile(File):
    def __init__(self, name, url):
        super().__init__(name)
        self.url = url

    def update(self):
        download_file(self.url, self.path)


class TaredFile(File):
    def __init__(self, name, entry, tarfile):
        super().__init__(name, [tarfile])
        self.entry = entry
        self.tarfile = tarfile

    def update(self):
        untar_file(self.tarfile.path, self.entry, self.path)


class ZippedFile(File):
    def __init__(self, name, zipfile, extract_all=False):
        super().__init__(name, [zipfile])
        self.zipfile = zipfile
        self.extract_all = extract_all

    def update(self):
        if self.extract_all:
            unzip_all_files(self.zipfile.path)
        else:
            unzip_file(self.zipfile.path, self.name)


class GzippedFile(File):
    def __init__(self, name, gzipfile):
        super().__init__(name, [gzipfile])
        self.gzipfile = gzipfile

    def update(self):
        gunzip_file(self.gzipfile.path, self.path)


class CSVFile(File):
    def __init__(self, name, csvfile, dtype=np.float32):
        super().__init__(name, [csvfile])
        self.csvfile = csvfile
        self.dtype = dtype

    def update(self):
        np.save(self.path, np.loadtxt(self.csvfile.path, dtype=self.dtype, delimiter=','))


class Dataset:
    def name(self):
        return type(self).__name__

    def get_sub_path(self):
        return self.name().lower()

    def get_path(self):
        with sub_path(self.get_sub_path()):
            return DATA_PATH

    def get_size(self):
        raise Exception('Not Implemented')

    def files(self, **kwargs):
        raise Exception('Not Implemented')

    def ensure_updated(self, min_mtime=0.):
        for f in self.files():
            f.ensure_updated(min_mtime)

    def get_data(self, **kwargs):
        self.ensure_updated()
        return self.get_data_raw(**kwargs)

    def get_data_raw(self, **kwargs):
        raise Exception('Not Implemented')

    def get_filenames(self):
        self.ensure_updated()
        return [f.path for f in self.files()]

    def parse_example(self, serialized_example):
        raise Exception('Not Implemented')

    def get_example_dtype(self):
        raise Exception('Not Implemented')

    def create_queue(self, epochs=None, read_batch_size=1024):
        """set read_batch_size to 0 if you only want single examples instead of batches"""
        filenames = self.get_filenames()
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs, shuffle=True)
        reader = tf.TFRecordReader()
        if read_batch_size == 0: # only read single examples
            _, serialized_example = reader.read(filename_queue)
            example = self.parse_example(serialized_example)
        else:
            _, serialized_examples = reader.read_up_to(filename_queue, read_batch_size)
            example = tf.map_fn(self.parse_example, serialized_examples, back_prop=False, dtype=self.get_example_dtype())
        return example


class LabeledImageMixin:
    def get_example_dtype(self):
        return (tf.float32, tf.int32)


if __name__ == '__main__':
    imdb = File('imdb.tar.gz')
    print(imdb.path)
    print(imdb.exists())
    print(imdb.last_modified())
    print(imdb.ensure_updated())
    try:
        print(imdb.ensure_updated(14571019940))
        print('This should not happen')
    except:
        pass
