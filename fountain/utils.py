import urllib.request
import tarfile
import gzip
import zipfile
import os
import os.path
import time
from PIL import Image
import numpy as np


def download_file(url, fn):
    urllib.request.urlretrieve(url, fn)


def touch(fn):
    now = time.time()
    os.utime(fn, (now, now))


def untar_file(fn, entry, path):
    with tarfile.open(fn) as tar:
        inf = tar.extractfile(entry)
        with open(path, 'wb') as f:
            f.write(inf.read())


def unzip_all_files(fn):
    with zipfile.ZipFile(fn) as zf:
        zf.extractall(os.path.dirname(fn))


def unzip_file(fn, entry):
    with zipfile.ZipFile(fn) as zf:
        zf.extract(entry, os.path.dirname(fn))


def gunzip_file(gzfn, path):
    with gzip.open(gzfn) as gzf:
        with open(path, 'wb') as f:
            for line in gzf:
                f.write(line)


class Embedder:
    def __init__(self, d, default):
        self.d = d
        self.default = default

    def embed(self, key):
        return self.d.get(key, self.default)


def jpg2npy(path):
    img = Image.open(path)
    return np.array(img)
