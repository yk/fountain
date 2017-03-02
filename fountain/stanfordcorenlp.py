#!/usr/bin/env python3

from fountain.data import *
import numpy as np
import _pickle as cPickle
from corenlp import StanfordCoreNLP


class CoreNLP(Dataset):
    def files(self):
        with sub_path('stanfordcorenlp'):
            jarfile = ZippedFile('stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar', OnlineFile('corenlp.zip', 'http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip'), extract_all=True)
            files = [jarfile]
            return files

    def get_data_raw(self):
        snlp = StanfordCoreNLP(corenlp_path=os.path.dirname(self.files()[0].path))
        return snlp

if __name__ == '__main__':
    cnlp = CoreNLP().get_data()
    print(cnlp.parse("Hi my name is Jonas"))
