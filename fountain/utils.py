import urllib.request
import tarfile
import gzip
import zipfile
import os
import os.path
import time
from PIL import Image
import numpy as np
import itertools as itt
import math


def download_file(url, fn):
    try:
        urllib.request.urlretrieve(url, fn)
    except Exception as e:
        print('Cannot retrieve', url)
        raise


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


def jpg2npy(path, resize=None, crop=None):
    img = Image.open(path)
    if crop:
        w, h = img.size
        wc = (w - crop[0]) / 2.
        hc = (h - crop[1]) / 2.
        l, r = math.floor(wc), w - math.ceil(wc)
        t, b = math.floor(hc), h - math.ceil(hc)
        img = img.crop((l, t, r, b))
    if resize:
        img = img.resize(resize, Image.ANTIALIAS)
    return np.array(img, dtype=np.uint8)


def to_chunks(iterable, chunk_size):
    it = iter(iterable)
    chunk = list(itt.islice(it, chunk_size))
    while(chunk):
        yield chunk
        chunk = list(itt.islice(it, chunk_size))


def get_chunk(iterable, chunk_size, chunk_number):
    return list(itt.islice(iterable, chunk_size * chunk_number, chunk_size * (chunk_number + 1)))


def create_batch_iterator(create_iter, repeats=-1, batch_size=1, buf_size=None, shuffle=False, trailing_elements=False):
    if buf_size is None:
        if shuffle:
            buf_size = batch_size * 10
        else:
            buf_size = batch_size

    assert buf_size >= batch_size

    buf = []
    idx = 0

    while idx != repeats:
        idx += 1
        for el in create_iter():
            buf.append(el)
            if len(buf) == buf_size:
                if shuffle:
                    random.shuffle(buf)
                batch = buf[:batch_size]
                buf = buf[batch_size:]
                yield batch

    while len(buf) >= batch_size:
        batch = buf[:batch_size]
        buf = buf[batch_size:]
        yield batch

    if trailing_elements:
        return buf

