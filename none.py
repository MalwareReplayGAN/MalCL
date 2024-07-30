import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
import joblib
from model import Generator, Discriminator, Classifier
from data_ import get_ember_train_data, extract_100data, oh, get_ember_test_data
from function2 import get_iter_train_dataset, get_iter_test_dataset, selector, test, get_dataloader
import math
import time
from torch.utils.data import TensorDataset

#
# parser = argparse.ArgumentParser('./main.py', description='Run')
# parser.add_argument('--data-dir', type=str)
# parser.add_arguement('--plot-dir', type=str, default='./plots')
# parser.add_arguement('--results-dir', type=str, default='./results')
# parser.add_argument('--saveddir', required=True, help='directory to save Generator and Classifier')

# experimental task parameters
# task_params = parser.add_argument_group('Task Parameters')
# task_params.add_argument('--target_classes', type=int, default=100, required=False, help='number of classes')
# task_params.add_argument('--init_classes', type=int, default=50, required=False, help='number of classes for the first task')
# task_params.add_argument('--data_set', type=str, choices=['drebin', 'EMBER'], help='dataset to use')

# training hyperparamters
# train_params = parser.add_argument_group('Training Parameters')
# train_params.add_argument('--lr', type=float, default=0.001, help='learning rate')
# train_params.add_argument('--epoch', type=int, default=30, help='number of training epochs')
# train_params.add_argument('--batch', type=int, default=256, help='batch-size')
# train_params.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='sgd')

# def create_parent_folder(file_path):
#    if not os.path.exists(os.path.dirname(file_path)):
#       os.makedirs(os.path.dirname(file_path))


#######
# GPU #
#######

# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"현재 사용 가능한 GPU의 수: {device_count}")
else:
    print("GPU를 사용할 수 없습니다.")

##############
# EMBER DATA #
##############

# Call the Ember Data

data_dir =  '/home/02mjpark/downloads/Continual_Learning_Malware_Datasets/EMBER_CL/EMBER_Class'
X_train, Y_train = get_ember_train_data(data_dir)
X_test, Y_test, Y_test_onehot = get_ember_test_data(data_dir)

feats_length= 2381
num_training_samples = 303331

#####################################
# Declarations and Hyper-parameters #
#####################################

seeds = [10, 20, 30]
init_classes = 50
final_classes = 100
n_inc = 5
nb_task = int(((final_classes - init_classes) / n_inc) + 1)
batchsize = 256
lr = 0.001
epoch_number = 50
z_dim = 62
k = 2
ls_a = []
momentum = 0.9
weight_decay = 0.000001


############################################
# data random arange #
#############################################


import random
import copy
import matplotlib.pyplot as plt

class_arr = np.arange(final_classes)
indices = torch.randperm(final_classes)
class_arr = torch.index_select(torch.Tensor(class_arr), dim=0, index=indices)

class_arr = list(class_arr)
Y_train_ = copy.deepcopy(Y_train)
Y_test_ = copy.deepcopy(Y_test)

for i in range(final_classes):
  Y_train[np.where(Y_train_ == class_arr[i])] = i
  Y_test[np.where(Y_test_ == class_arr[i])] = i

print("class_arr")
print(class_arr)




##########
# Models #
##########


C = Classifier()
C.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C.to(device)

'''
if use_cuda:
    G.cuda(0)
    D.cuda(0)
    C.cuda(0)
'''

C_optimizer = optim.SGD(C.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

#############
# Functions # 
#############


def run_batch(C, C_optimizer, x_, y_):
      x_ = x_.view([-1, feats_length])

      x_ = Variable(x_)
      
      if use_cuda:
        x_, y_ = x_.to(device), y_.to(device)

      # update C

      C_optimizer.zero_grad()
      # print("y_ shape", y_.shape) # 16
      output = C(x_)
      if use_cuda:
         output = output.to(device)

      C_loss = criterion(output, y_)

      C_loss.backward()
      C_optimizer.step()

      return output, C_loss


scaler = StandardScaler()

for task in range(nb_task):
  n_class = init_classes + task * n_inc
  # Load data for the current task
  X_train_t, Y_train_t = get_iter_train_dataset(X_train,  Y_train, n_class=n_class, n_inc=n_inc, task=task)
  nb_batch = int(len(X_train_t)/batchsize)
  print("nb_batch", nb_batch)
  train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=batchsize, n_class=n_class, scaler = scaler)
  X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class=n_class)

  if task > 0:
    C = C.expand_output_layer(init_classes, n_inc, task)
    C = C
    C.to(device)

  for epoch in range(epoch_number):
    train_loss = 0.0
    train_acc = 0.0
    
    for n, (inputs, labels) in enumerate(train_loader):
      
      inputs = inputs.float()
      labels = labels.float()
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs, loss = run_batch(C,C_optimizer, inputs, labels)

      train_loss += loss.item() * inputs.size(0) # calculate training loss and accuracy
      _, preds = torch.max(outputs, 1)
      class_label = torch.argmax(labels.data, dim=-1)
      train_acc += torch.sum(preds == class_label)

      nb_per_10 = int(nb_batch/10)

    print("epoch:", epoch+1)
    train_loss = train_loss / len(X_train_t)
    train_acc = float(train_acc / len(X_train_t))
    print("train_loss: ", train_loss)
    print("train_acc: ", train_acc)

  # test

  with torch.no_grad():
    accuracy = test(model=C, x_test=X_test_t, y_test=Y_test_t, n_class=n_class, device = device, scaler = scaler)
    if epoch == epoch_number-1:
      ls_a.append(accuracy)

  print("task", task, "done")

  if task == nb_task-1:
      print("The Accuracy for each task:", ls_a)
      print("The Global Average:", sum(ls_a)/len(ls_a))
