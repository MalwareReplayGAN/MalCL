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
from drebin_model_FML import Generator, Discriminator, Classifier
from drebin_data_ import get_ember_train_data, extract_100data, oh, get_ember_test_data
from function import get_iter_train_dataset, get_iter_test_dataset, selector, test, get_dataloader
import math
import time
from torch.utils.data import TensorDataset


#######
# GPU #
#######

# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(10)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"현재 사용 가능한 GPU의 수: {device_count}")
else:
    print("GPU를 사용할 수 없습니다.")

##############
# EMBER DATA #
##############

# Call the Ember Data

data_dir =  '/home/02mjpark/downloads/AZ-Class-with-Drebin-Features/AZ-Class-with-Drebin-Features'
X_train, Y_train = get_ember_train_data(data_dir)
X_test, Y_test, Y_test_onehot = get_ember_test_data(data_dir)

feats_length= 2439
num_training_samples = 257023

#####################################
# Declarations and Hyper-parameters #
#####################################

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
#################################################

class_arr = np.arange(final_classes)
indices = torch.randperm(final_classes)
class_arr = torch.index_select(torch.Tensor(class_arr), dim=0, index=indices)

class_arr = np.array(class_arr)
class_arr = list(class_arr)
#print("np.unique(class_arr)")
#print(np.unique(class_arr))

Y_train_ = copy.deepcopy(Y_train)
Y_test_ = copy.deepcopy(Y_test)

for i in range(0, final_classes):
  Y_train[np.where(Y_train_ == class_arr[i])] = i
  Y_test[np.where(Y_test_ == class_arr[i])] = i




##########
# Models #
##########

G = Generator()
D = Discriminator()
C = Classifier()

G.train()
D.train()
C.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G.to(device)
D.to(device)
C.to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
C_optimizer = optim.SGD(C.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

#############
# Functions # 
#############

def run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, x_, y_):
      x_ = x_.view([-1, feats_length])

      y_real_ = Variable(torch.ones(x_.size(0), 1))
      y_fake_ = Variable(torch.zeros(x_.size(0), 1))

      if use_cuda:
        y_real_, y_fake_ = y_real_.to(device), y_fake_.to(device)

      z_ = torch.rand((x_.size(0), z_dim))

      x_, z_ = Variable(x_), Variable(z_)

      if use_cuda:
        x_, z_, y_ = x_.to(device), z_.to(device), y_.to(device)

      # update D network
      D_optimizer.zero_grad()

      D_real, _ = D(x_)
      D_real_loss = BCELoss(D_real, y_real_[:x_.size(0)])

      G_ = G(z_)
      D_fake, _ = D(G_)
      D_fake_loss = BCELoss(D_fake, y_fake_[:x_.size(0)])

      D_loss = D_real_loss + D_fake_loss

      D_loss.backward()
      D_optimizer.step()

      # update G network
      G_optimizer.zero_grad()

      fake_data = G(z_)
      _, features_fake = D(fake_data)
      _, features_real_unl = D(x_)
      feature_mean_real = torch.mean(features_real_unl, dim=0)
      feature_mean_fake = torch.mean(features_fake, dim=0)
      G_loss = torch.mean(torch.abs(feature_mean_real - feature_mean_fake))

      G_loss.backward()
      G_optimizer.step()

      # update C

      C_optimizer.zero_grad()
      output = C(x_)
      if use_cuda:
         output = output.to(device)

      C_loss = criterion(output, y_)

      C_loss.backward()
      C_optimizer.step()

      return output, C_loss


def get_replay_with_label(generator, classifier, batchsize, n_class, task, logits_arr):
  arr = []
  z_ = Variable(torch.rand((batchsize, z_dim)))
  if use_cuda:
    z_ = z_.to(device)

  images = generator(z_)

  label = classifier.predict(images)
  logits = classifier.get_logits(images)
  if use_cuda:
     logits = logits.to(device)


  for (log_gen, log) in zip(logits, logits_arr):
    arr.append(torch.mean(torch.abs(log - log_gen)))

  arr = torch.Tensor(arr).to(device)

  if batchsize<(n_class-5)*2:
      sample_num = batchsize
  else: sample_num = (n_class-5)*2

  for_one_hot = torch.Tensor([list(i).index(max(i)) for i in label[arr.sort(0)[1][:sample_num]]])

  return images[arr.sort(0)[1][:sample_num]].to(device), nn.functional.one_hot(for_one_hot.to(torch.int64), num_classes = n_class).to(device)


# We reinit D and G to not cheat
G.reinit()
D.reinit()

logits_arr = []
logits_arr_collect = []

scaler = StandardScaler()

for task in range(nb_task):

  n_class = init_classes + task * n_inc

  X_train_t, Y_train_t = get_iter_train_dataset(X_train,  Y_train, n_class=n_class, n_inc=n_inc, task=task)
  nb_batch = int(len(X_train_t)/batchsize)
  
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
        
      if task > 0 :
          replay, re_label = get_replay_with_label(G_saved, C_saved, batchsize, n_class=n_class, task=task, logits_arr = logits_arr)

          inputs=torch.cat((inputs,replay),0)
          labels=torch.cat((labels,re_label),0) 

      outputs, loss = run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, inputs, labels)
      train_loss += loss.item() * inputs.size(0) # calculate training loss and accuracy
      _, preds = torch.max(outputs, 1)

      class_label = torch.argmax(labels.data, dim=-1)

      train_acc += torch.sum(preds == class_label)
      if epoch == epoch_number-1:
        logits_arr_collect.append(C.get_logits(inputs))
      print("\r", task, "task", epoch+1, "epoch", n, "batch", end="")

    print("\n")
    print("epoch:", epoch+1)
    train_loss = train_loss / len(X_train_t)
    train_acc = float(train_acc / len(X_train_t))
    print("train_loss: ", train_loss)
    print("train_acc: ", train_acc)

  G_saved = deepcopy(G)
  C_saved = deepcopy(C)

  logits_arr = torch.mean(torch.stack(logits_arr_collect, dim=0), dim=0)
  logits_arr_collect = []

  # test
    
  with torch.no_grad():
      accuracy = test(model=C_saved, x_test=X_test_t, y_test=Y_test_t, n_class=n_class, device = device, scaler = scaler)
      ls_a.append(accuracy)
  print("task", task, "done")

  if task == nb_task-1:
     print("The Accuracy for each task:", ls_a)
     print("The Global Average:", sum(ls_a)/len(ls_a))
