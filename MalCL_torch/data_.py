
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

    return X_test, Y_test

def shuffle_data(x_, y_, s):
    random.seed(s)
    indices = list(range(len(x_)))
    random.shuffle(indices)
    x_ = x_[indices]
    y_ = y_[indices]
    return x_, y_

def oh(Y, num_classes):
    Y = torch.FloatTensor(Y)
    Y_oh = nn.functional.one_hot(Y.to(torch.int64), num_classes=num_classes)
    return Y_oh
