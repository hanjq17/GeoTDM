import os
import pickle


class TrajDataset(object):
    def __init__(self, root, name, force_reprocess):
        self.root = root
        self.name = name
        self.force_reprocess = force_reprocess
        self.processed_path = self.processed_file()
        if os.path.exists(self.processed_path) and not force_reprocess:
            print(f'Dataset {self.name} found at {self.processed_path}')
        else:
            if not os.path.exists(self.processed_path):
                print(f'Dataset {self.name} not found at {self.processed_path}')
            else:
                print(f'Dataset {self.name} found at {self.processed_path} but forced to reprocess')
            print(f'Processing data')
            self.preprocess_raw()
        print('Loading data')
        with open(self.processed_path, 'rb') as f:
            self.data = pickle.load(f)
        print('Data loaded')
        self.postprocess()
        print('Data post-processed')

    def processed_file(self):
        return os.path.join(self.root, self.name + '.pt')

    def preprocess_raw(self):
        raise NotImplementedError()

    def postprocess(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()
