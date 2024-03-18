<<<<<<< HEAD
import numpy as np
import torch
import torch.nn as nn

def get_ember_train_data(data_dir):

    XY_train = np.load(data_dir + '/XY_train.npz')
    X_train, Y_train = XY_train['X_train'], XY_train['Y_train']
    Y_train = torch.LongTensor(Y_train)
    Y_train_onehot = nn.functional.one_hot(Y_train, num_classes=100)
    return X_train, Y_train, Y_train_onehot

def get_ember_test_data(data_dir):
    XY_test = np.load(data_dir + '/XY_test.npz')
    X_test, Y_test = XY_test['X_test'], XY_test['Y_test']
    Y_test = torch.LongTensor(Y_test)
    Y_test_onehot = nn.functional.one_hot(Y_test, num_classes=100)

    return X_test, Y_test, Y_test_onehot

# 각 레이블 당 100개씩만 데이터 뽑기

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
=======
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
>>>>>>> 7cacb0c84a28d4f7d18f1547c8256609b0e74c19

def extract_100data(X_train, Y_train):

    labels = np.unique(Y_train)

    selected_indices = []
    for label in labels:
        label_indices = np.where(Y_train == label)[0]
        selected_indices.extend(label_indices[:100])
    selected_indices = np.array(selected_indices)
    X_train_s = X_train[selected_indices]
    Y_train_s = Y_train[selected_indices]
    Y_train_s = torch.LongTensor(Y_train_s)
    Y_train_s_oh = nn.functional.one_hot(Y_train_s)
    # print("X_train_s", X_train_s.shape)
    # unique, counts = np.unique(Y_selected, return_counts=True)
    # print(unique, counts)
    # print("size of Y_train_100", Y_selected.shape)
    return X_train_s, Y_train_s, Y_train_s_oh

# X_train, Y_train, X_test, Y_test, Y_train_onehot, Y_test_onehot = get_ember_data(data_dir)
# print("X_train.shape", X_train.shape)
# print("Y_train", Y_train)
# print("X_test.shape", X_test.shape)
# print("Y_test", Y_test)
# print("Y_train_onehot", Y_train_onehot)
# print("Y_train_onehot", Y_train_onehot.shape)
# print("Y_test_onehot", Y_test_onehot)
