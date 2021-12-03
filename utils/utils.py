from torch.utils.data import Dataset
from torchtext.legacy import data
from sklearn.model_selection import KFold
import numpy as np
import random

class TextDataset(data.Dataset):
    """ 
    Code modified from: 
    https://gist.github.com/lextoumbourou/8f90313cbc3598ffbabeeaa1741a11c8
    """
    def __init__(self, ds, fields, is_test=False, **kwargs):
        examples = []
        try:
            for text, label in ds:
                label = None if is_test else label

                examples.append(data.Example.fromlist([text, label], fields))
        except:
            examples = ds
 
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_ds, val_ds=None, test_ds=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_ds is not None:
            train_data = cls(train_ds.copy(), data_field, **kwargs)
        if val_ds is not None:
            val_data = cls(val_ds.copy(), data_field, **kwargs)
        if test_ds is not None:
            test_data = cls(test_ds.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

class BasicDataset(Dataset):
    def __init__(self, data_path) -> None:
        self.data = []
        self.label = []
        with open(data_path, encoding="ascii", errors="ignore") as data:
            for line in data:
                true, false = line.split('\t')
                false = false[:-1]
                if random.random() > 0.5:
                    self.data.append((true + false).lower())
                    self.label.append(0)
                else:
                    self.data.append((false + true).lower())
                    self.label.append(1)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def get_pairs(self):
        return [(self.data[i], self.label[i]) for i in range(len(self.data))]
    
    def __len__(self):
        return len(self.data)

class TestDataset(Dataset):
    def __init__(self, data_path) -> None:
        self.data = []
        self.label = []
        with open(data_path, encoding="ascii", errors="ignore") as data:
            for line in data:
                true, false = line.split('\t')
                false = false[:-1]
                self.data.append((true + false).lower())
                self.label.append(0)

    def __getitem__(self, index):
        return self.data[index]

    def get_pairs(self):
        return [(self.data[i], self.label[i]) for i in range(len(self.data))]
    
    def __len__(self):
        return len(self.data)
class TextGenDataset(Dataset):
    def __init__(self, data_path) -> None:
        self.data = []
        self.label = []
        with open(data_path, encoding="ascii", errors="ignore") as data:
            for line in data:
                true, false = line.split('\t')
                false = false[:-1]
                self.data.append(true)
                self.label.append(false)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def get_pairs(self):
        return [(self.data[i], self.label[i]) for i in range(len(self.data))]
    
    def __len__(self):
        return len(self.data)


def get_fold_data(train_ds, fields, num_folds=10):
    """
    More details about 'fields' are available at 
    https://github.com/pytorch/text/blob/master/torchtext/datasets/imdb.py
    """
    
    kf = KFold(n_splits=num_folds, shuffle=True)
    train_data_arr = np.array(train_ds.examples)

    for train_index, val_index in kf.split(train_data_arr):
        yield(
            TextDataset(train_data_arr[train_index], fields=fields),
            TextDataset(train_data_arr[val_index], fields=fields),
        )

def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def tokenizer(word):
    return list(word)


