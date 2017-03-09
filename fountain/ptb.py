#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import itertools as itt
from fountain.stanfordcorenlp import CoreNLPParsedFile


class PTB(Dataset):
    def __init__(self, vocabularize=False, parse=False):
        super().__init__()
        self.vocabularize = vocabularize
        self.parse = parse

    def files(self):
        with sub_path('ptb'):
            filenames = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']
            files = [OnlineFile(f, 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/{}'.format(f)) for f in filenames]
            if self.parse:
                pfiles = [CoreNLPParsedFile(f[:-3] + '.parsed', [f]) for f in files]
                files.extend(pfiles)
            return files

    def get_data_raw(self):
        results = []
        sets = []
        files = self.files()
        if self.parse:
            files, pfiles = files[:3], files[3:]
        for f in files:
            with open(f.path) as f:
                lines = [l.strip() for l in f.readlines()]
                sets.append(lines)
        results.append(sets)
        
        if self.vocabularize:
            splitsets = [[l.split() for l in s] for s in sets]
            vocab = [(w, i) for i, w in enumerate(set(itt.chain(*itt.chain(*splitsets))))]
            vocabf = dict(vocab)
            vocabb = dict([(i, w) for w, i in vocab])
            idsets = []
            for s in splitsets:
                idsets.append([])
                for l in s:
                    idsets[-1].append(np.array([vocabf[w] for w in l], np.int32))

            results.append((vocabf, vocabb), idsets)
        
        if self.parse:
            psets = []
            for f in pfiles:
                with open(f.path) as f:
                    plines = [l.strip() for l in f]
                    psets.append(plines)
            results.append(psets)
                    


        return results

if __name__ == '__main__':
    print(PTB().get_data())
