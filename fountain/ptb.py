#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import itertools as itt


class PTB(Dataset):
    def files(self):
        with sub_path('ptb'):
            filenames = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']
            files = [OnlineFile(f, 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/{}'.format(f)) for f in filenames]
            return files

    def get_data_raw(self, only_lines=False):
        sets = []
        for f in self.files():
            with open(f.path) as f:
                lines = [l.strip().split() for l in f.readlines()]
                sets.append(lines)
        if only_lines:
            return [[" ".join(l) for l in s] for s in sets]
        vocab = [(w, i) for i, w in enumerate(set(itt.chain(*itt.chain(*sets))))]
        vocabf = dict(vocab)
        vocabb = dict([(i, w) for w, i in vocab])
        idsets = []
        for s in sets:
            idsets.append([])
            for l in s:
                idsets[-1].append(np.array([vocabf[w] for w in l], np.int32))

        return (vocabf, vocabb), idsets

if __name__ == '__main__':
    print(PTB().get_data())
