#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle
from corenlp import StanfordCoreNLP
from tqdm import tqdm


class CoreNLP(Dataset):
    def files(self):
        with sub_path('stanfordcorenlp', relative=False):
            jarfile = ZippedFile('stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar', OnlineFile('corenlp.zip', 'http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip'), extract_all=True)
            files = [jarfile]
            return files

    def get_data_raw(self):
        snlp = StanfordCoreNLP(corenlp_path=os.path.dirname(self.files()[0].path))
        return snlp


class CoreNLPParsedFile(File):
    def __init__(self, name, dependencies):
        super().__init__(name, dependencies)
        CoreNLP().ensure_updated()

    def update(self):
        cnlp = CoreNLP().get_data_raw()
        data = []
        for b in self.dependencies:
            with open(b.path) as f:
                with open(self.path, 'w') as rf:
                    for l in tqdm(f):
                        r = cnlp.parse(l.strip())
                        rf.write(r + '\n')


if __name__ == '__main__':
    cnlp = CoreNLP().get_data()
    print(cnlp.parse("Hi my name is Jonas"))
