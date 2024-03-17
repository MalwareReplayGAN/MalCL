import copy
import numpy as np
from sklearn.utils import shuffle
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch
from sklearn.preprocessing import StandardScaler

def get_ember_data(data_dir):

    XY_train = np.load(data_dir + '/XY_train.npz')
    X_train, Y_train = XY_train['X_train'], XY_train['Y_train']

    XY_test = np.load(data_dir + '/XY_test.npz')
    X_test, Y_test = XY_test['X_test'], XY_test['Y_test']

    return X_train, Y_train, X_test, Y_test

