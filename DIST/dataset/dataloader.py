# dataloader.py

import pickle
import numpy as np
import os
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse


def max_min_normalization(x, _max, _min):
    r"""
    Max-min normalization

    _max: float
        Max
    _min: float
        Min
    """
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    r"""
    Max-min re-normalization

    _max: float
        Max
    _min: float
        Min
    """
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()
    
    def get_len(self):
        return self.num_batch
    
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
    
def load_dataset(data_dir, dataset, n_obs, batch_size):
    data = {}

    for category in ['train', 'val', 'test']:
        cat_data = np.load(
            os.path.join(data_dir, category + '.npz')) 
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

        if n_obs is not None:
            data['x_' + category] = data['x_' + category][:n_obs]
            data['y_' + category] = data['y_' + category][:n_obs]

    if dataset == 'METR-LA' or dataset == 'bay':
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        for category in ['train', 'val', 'test']: 
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size)
        data['scaler'] = scaler
        data['_min'] = None
        data['_max'] = None
    elif dataset == 'PEMS08' or dataset == 'PEMS04' or dataset == 'PEMS07':
        _min = pickle.load(open(data_dir + "/min.pkl", 'rb'))
        _max = pickle.load(open(data_dir + "/max.pkl", 'rb'))

        data['train_loader']   = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader']     = DataLoader(data['x_val'], data['y_val'], batch_size)
        data['test_loader']    = DataLoader(data['x_test'], data['y_test'], batch_size)
        data['scaler']         = re_max_min_normalization
        data['_min'] = _min
        data['_max'] = _max

    return data

def load_graph(data_dir, dataset):
    pkl_filename = os.path.join(data_dir, dataset, 'adj_mx.pkl')
    if dataset == 'METR-LA' or dataset == 'bay':
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    elif dataset == 'PEMS08' or dataset == 'PEMS04' or dataset == 'PEMS07':
        adj_mx = load_pickle(pkl_filename)
    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj_mx))

    return edge_index, edge_attr, adj_mx