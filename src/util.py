import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import from_numpy
from collections import namedtuple
import pandas as pd


Action = namedtuple('Action', ['position', 'action'])

Perspective = namedtuple('Perspective', ['perspective', 'position'])

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'terminal'])


def conv_to_fully_connected(input_size, filter_size, padding, stride):
    return (input_size - filter_size + 2 * padding)/ stride + 1


def pad_circular(x, pad):
    x = torch.cat([x, x[:,:,:,0:pad]], dim=3)
    x = torch.cat([x, x[:,:, 0:pad,:]], dim=2)
    x = torch.cat([x[:,:,:,-2 * pad:-pad], x], dim=3)
    x = torch.cat([x[:,:, -2 * pad:-pad,:], x], dim=2)
    return x


def incremental_mean(x, mu, N):
    return mu + (x - mu) / (N)


def convert_from_np_to_tensor(tensor):
    tensor = from_numpy(tensor)
    tensor = tensor.type('torch.Tensor')
    return tensor

class MCMCDataReader:  # This is the object we crate to read a file during training
    def __init__(self, file_path, size):
        self.__file_path = file_path
        self.__size = size
        self.__df = pd.read_pickle(file_path)
        self.__current_index = 0
        self.__capacity = self.__df.index[-1][0] + 1  # The number of data samples in the dataset

    def next(self):
        if self.__current_index < self.__capacity:
            qubit_matrix = self.__df.loc[self.__current_index, 0:1, :, :].to_numpy(copy=True).reshape((2, self.__size, self.__size))
            eq_distr = self.__df.loc[self.__current_index, 2:17, 0, 0].to_numpy(copy=True).reshape((-1))  # kanske inte behöver kopiera här?
            self.__current_index += 1
            return qubit_matrix, eq_distr
        else:
            return None, None  # How do we do this nicely? Maybe it can wrap around?
    
    def has_next(self):
        return self.__current_index < self.__capacity

    def current_index(self):
        return self.__current_index