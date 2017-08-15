from fountain.data import *
import numpy as np
from hashlib import md5
import pickle
from gensim.models.keyedvectors import KeyedVectors, Vocab


class WordVectorsFile(OnlineFile):
    def __init__(self, name, source_file_url, limit=None):
        super().__init__(name, source_file_url)
        self.limit = limit
        self._wv = None

    def get_word_vectors(self):
        if not self._wv:
            self._wv = KeyedVectors.load_word2vec_format(self.path, limit=self.limit)
        return self._wv


class EmbeddingsFile(File):
    def __init__(self, name, word_vectors_file, vocabs_file):
        super().__init__(name, [word_vectors_file, vocabs_file])
        self.word_vectors_file = word_vectors_file
        self.vocabs_file = vocabs_file
        self._embs = None

    def get_embeddings(self):
        if not self._embs:
            self._embs = KeyedVectors.load(self.path)
        return self._embs

    def update(self):
        wv = self.word_vectors_file.get_word_vectors()
        voc = self.vocabs_file.get_vocabs()['word']
        words_in_vocab = [k for k, _ in sorted(voc.items(), key=lambda i: i[1][0])]
        word_embs = wv[words_in_vocab[1:]]
        unk_emb = np.mean(word_embs, 0, keepdims=True)
        embs = np.concatenate((unk_emb, word_embs), 0)
        kv = KeyedVectors()
        kv.syn0 = embs
        kv.vocab = dict((k, Vocab(index=v[0], count=v[1])) for k, v in voc.items())
        kv.index2word = words_in_vocab
        kv.save(self.path)



class ConllDataFile(OnlineFile):
    def get_iterator(self, start_at=0, min_sentence_length=2, max_sentence_length=-1):
        with open(self.path) as f:
            idx = 0
            block, actions = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if len(block) > 0:
                        if len(block) >= min_sentence_length and (len(block) <= max_sentence_length or max_sentence_length < 0):
                            idx += 1
                            yield block, actions
                        block, actions = [], []
                elif idx < start_at:
                    continue
                elif line.startswith('T'):
                    actions = []
                    for a in line.split()[2::2]:
                        if a == 'SHIFT':
                            actions.append(dict(type='SHIFT', label=None))
                        else:
                            a1, a2 = a.split('(')
                            a2 = a2[:-1].lower()
                            actions.append(dict(type=a1, label=a2))
                else:
                    widx, word, _, category, tag, morph, head, label, _, _ = map(lambda s: '' if s == '_' else s, line.split())
                    head = int(head) - 1
                    widx = int(widx) - 1
                    morph = morph.lower().split('|')
                    token = dict(word=word.lower(), category=category.lower(), tag=tag.lower(), morph=morph, head=head, idx=widx, label=label.lower())
                    block.append(token)
            return block, actions

TOKEN_ATTRS = ('word', 'category', 'tag', 'label')
MORPH_ATTRS = tuple(map(str.lower, ('fPOS', 'NumType', 'Number', 'Case', 'Gender', 'Person', 'PronType', 'Mood', 'Tense', 'VerbForm', 'Degree', 'Definite', 'Poss', 'Voice', 'Reflex')))

ATTRS = TOKEN_ATTRS + MORPH_ATTRS

class VocabsFile(File):
    def __init__(self, name, data_file, word_vectors_file, truncate_words=5):
        super().__init__(name, [data_file, word_vectors_file])
        self.data_file = data_file
        self.word_vectors_file = word_vectors_file
        self.truncate_words = truncate_words
        self._vocs = None

    def get_vocabs(self):
        if not self._vocs:
            with open(self.path, 'rb') as f:
                self._vocs = pickle.load(f)
        return self._vocs

    def update(self):
        known_words = set(self.word_vectors_file.get_word_vectors().vocab.keys())
        vocabs = dict((n, {}) for n in TOKEN_ATTRS + MORPH_ATTRS)
        for block, actions in self.data_file.get_iterator():
            for tok in block:
                for a in TOKEN_ATTRS:
                    voc = vocabs[a]
                    ta = tok[a]
                    if a == 'word' and ta not in known_words:
                        continue
                    if ta not in voc:
                        voc[ta] = 0
                    voc[ta] += 1
                for m in tok['morph']:
                    ma, mv = m.split('=')
                    voc = vocabs[ma]
                    if ma not in voc:
                        voc[mv] = 0
                    voc[mv] += 1

        ctvocabs = dict()
        for k, v in vocabs.items():
            ctvd = dict()
            unks = 0
            for idx, (kk, ct) in enumerate(sorted(v.items(), key=lambda i: i[1], reverse=True)):
                if k == 'word' and ct < self.truncate_words:
                    unks += ct
                    continue
                ctvd[kk] = (idx+1, ct)
            ctvd['<UNK>'] = (0, unks)
            ctvocabs[k] = ctvd
            vocabs = ctvocabs
        with open(self.path, 'wb') as f:
            pickle.dump(vocabs, f)



class ConllDataset(Dataset):
    def __init__(self, name, data_url, word_vectors_url, start_at=0, limit=-1, words_limit=100000, min_sentence_length=2, max_sentence_length=-1):
        super().__init__()
        self._name = name
        self.data_url = data_url
        self.word_vectors_url = word_vectors_url
        self.start_at = start_at
        self.limit = limit
        self.words_limit = words_limit
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length

    def name(self):
        return self._name

    def get_size(self):
        return self.limit

    def files(self):
        with sub_path('word_vectors'):
            wvname = self.word_vectors_url.split('/')[-1]
            wvfile = WordVectorsFile(wvname, self.word_vectors_url, limit=self.words_limit)
        with sub_path(self.get_sub_path()):
            data_file = ConllDataFile('{}.parse'.format(self._name), self.data_url)
            voc_file = VocabsFile('{}.vocabs'.format(self._name), data_file, wvfile)
            emb_file = EmbeddingsFile('{}.embs'.format(self._name), wvfile, voc_file)
            files = [emb_file, voc_file, data_file]
            return files

    def get_embeddings(self):
        self.ensure_updated()
        return self.files()[0].get_embeddings()

    def get_vocabs(self):
        self.ensure_updated()
        return self.files()[1].get_vocabs()

    def create_iterator(self):
        vocabs = self.files()[1].get_vocabs()
        lvoc, wvoc = vocabs['label'], vocabs['word']
        num_labels = len(lvoc)
        num_words = len(wvoc)
        word_start_idx = 2 * num_labels

        data = []
        for rdidx, (block, actions) in enumerate(self.files()[2].get_iterator(start_at=self.start_at, min_sentence_length=self.min_sentence_length, max_sentence_length=self.max_sentence_length)):
            if rdidx == self.limit:
                break
            ls = []
            for tok in block:
                l = []
                for a in TOKEN_ATTRS:
                    s = tok[a]
                    sid, sct = vocabs[a].get(s, (0, 0))
                    if a == 'label':
                        label = sid
                    l.append(sid)
                morph = dict([m.split('=') for m in tok['morph']])
                for a in MORPH_ATTRS:
                    if a in morph:
                        mid, mct = vocabs[a][morph[a]]
                    else:
                        mid = 0
                    l.append(mid)
                ls.append((l, tok['idx'], tok['head'], label, tok))

            shift_idx = 0
            als = []
            for a in actions:
                if a['type'] == 'SHIFT':
                    t = block[shift_idx]
                    wid, wct = wvoc.get(t['word'], (0, 0))
                    albl = word_start_idx + wid
                    idx = shift_idx
                    shift_idx += 1
                else:
                    lid, lct = lvoc[a['label']]
                    albl = 2 * lid
                    if a['type'] == 'RIGHT_ARC':
                        albl += 1
                    idx = -1
                als.append((albl, a, idx))

            assert shift_idx == len(block)

            yield ls, als


if __name__ == '__main__':
    ds = ConllDataset('fr_en.en', 'http://cake.da.inf.ethz.ch:8080/fr_en.en.parse', 'http://cake.da.inf.ethz.ch:8080/wiki.en.vec', limit=1000, words_limit=10000)
    print(ds.get_embeddings())
    print(next(ds.create_iterator()))
