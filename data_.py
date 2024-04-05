
import numpy as np
import torch
import torch.nn as nn
import random

def get_ember_train_data(data_dir):

    XY_train = np.load(data_dir + '/XY_train.npz')
    X_train, Y_train = XY_train['X_train'], XY_train['Y_train']

    return X_train, Y_train

def get_ember_test_data(data_dir):
    XY_test = np.load(data_dir + '/XY_test.npz')
    X_test, Y_test = XY_test['X_test'], XY_test['Y_test']
    Y_test = torch.LongTensor(Y_test)
    Y_test_onehot = nn.functional.one_hot(Y_test, num_classes=100)
    return X_test, Y_test, Y_test_onehot

def shuffle_data(x_, y_):
    indices = list(range(len(x_)))
    random.shuffle(indices)
    x_ = x_[indices]
    y_ = y_[indices]
    return x_, y_

def oh(Y):
    Y = torch.LongTensor(Y)
    Y_oh = nn.functional.one_hot(Y)
    return Y_oh

def extract_100data(X_train, Y_train):

    labels = np.unique(Y_train)

    selected_indices = []
    for label in labels:
        label_indices = np.where(Y_train == label)[0]
        selected_indices.extend(label_indices[:100])
    selected_indices = np.array(selected_indices)
    X_train_s = X_train[selected_indices]
    Y_train_s = Y_train[selected_indices]
    # unique, counts = np.unique(Y_selected, return_counts=True)
    # print(unique, counts)
    # print("size of Y_train_100", Y_selected.shape)
    return X_train_s, Y_train_s

