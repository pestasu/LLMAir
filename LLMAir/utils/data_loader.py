import os
import datetime
import ipdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings

warnings.filterwarnings('ignore')

class StandardScaler():
    """
    Standard scaler for input normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class get_dataset(Dataset):
    def __init__(self, args, mode='train'):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.data_root_path = args.data_root_path
        self.data_path = args.data_path

        self.__read_data__(mode) 

    def __read_data__(self, mode):
        data = np.load(os.path.join(self.data_root_path, self.data_path, mode+'.npz'))
        self.x = data['x'].transpose(1, 0, 2, 3) # num_samples, num_station, seq_len, num_features
        self.y = data['y'].transpose(1, 0, 2, 3)
        if mode == 'train':
            self.scaler = data['scaler']
        
        
    def __getitem__(self, index):
        return torch.Tensor(self.x[index]), torch.Tensor(self.y[index])

    def __len__(self):
        return len(self.x) - self.seq_len - self.pred_len + 1

class get_dataset_time(Dataset):
    def __init__(self, args, mode='train'):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.data_root_path = args.data_root_path
        self.data_path = args.data_path

        self.__read_data__(mode) 

    def __read_data__(self, mode):
        data = np.load(os.path.join(self.data_root_path, self.data_path, mode+'.npz'))
        num_station, num_samples, seq_len, num_features = data['x'].shape
        self.x = data['x'].reshape(num_station * num_samples, seq_len, num_features)
        self.y = data['y'].reshape(num_station * num_samples, -1, num_features)
        if mode == 'train':
            self.scaler = data['scaler']
        # ipdb.set_trace()
        
    def __getitem__(self, index):
        return torch.Tensor(self.x[index]), torch.Tensor(self.y[index])

    def __len__(self):
        return len(self.x) - self.seq_len - self.pred_len + 1

def get_dataloader(args, need_location=True):

    get_dataset_fuc = get_dataset if need_location else get_dataset_time
    
    datasets = {
        'train': get_dataset_fuc(args, mode='train'),
        'valid': get_dataset_fuc(args, mode='valid'),
        'test': get_dataset_fuc(args, mode='test')
    }

    scalers = datasets['train'].scaler
    dataLoader = {
        ds: DataLoader(datasets[ds],
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False if ds == 'test' else True)
        for ds in datasets.keys()
    }
    return dataLoader, scalers