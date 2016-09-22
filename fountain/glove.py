from data import *
import numpy as np
import pickle


class Glove6B(Dataset):
    def __init__(self, dimensions=50):
        super().__init__()
        self.dimensions = dimensions

    def files(self):
        with sub_path('glove'):
            zipFile = OnlineFile('glove.6B.zip', 'http://nlp.stanford.edu/data/glove.6B.zip')
            txtFile = ZippedFile('glove.6B.{}d.txt'.format(self.dimensions), zipFile)
            dict_file = self.GloveDataFile(txtFile.name[:-3] + "pkl", self.dimensions, txtFile)
            return [dict_file]

    def get_data_raw(self):
        dict_file = self.files()[0]
        with open(dict_file.path, 'rb') as f:
            d = pickle.load(f)
        return Embedder(d, np.zeros(self.dimensions, dtype=np.float32))

    class GloveDataFile(File):
        def __init__(self, name, dimensions, txtFile):
            super().__init__(name, [txtFile])
            self.dimensions = dimensions
            self.txtFile = txtFile

        def update(self):
            d = dict()
            with open(self.txtFile.path) as f:
                for line in f:
                    tokens = line.strip().split()
                    d[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
            with open(self.path, 'wb') as f:
                pickle.dump(d, f)


if __name__ == '__main__':
    print(Glove6B().get_data().embed('dog').shape)
    print(Glove6B(100).get_data().embed('dog').shape)
