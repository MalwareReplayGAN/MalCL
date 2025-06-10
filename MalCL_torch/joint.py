import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
import joblib
from models import Classifier
from function import test, class_pick_rand

from torch.utils.data import TensorDataset
from dynaconf import Dynaconf
from arguments import _parse_args
from setting import configurate, torch_setting
from _data import dataset
from train import data_task_joint, report_result
import subprocess
import random

# global variables and setting
config = Dynaconf()
args = _parse_args()
configurate(args, config)
torch_setting(config)


##############
# EMBER DATA #
##############

X_train, Y_train, X_test, Y_test = dataset(config)

# ############################################
# # data random arange #
# #############################################

Y_train, Y_test = class_pick_rand(config, Y_train, Y_test)


###############################
# Models and Hyper Parameters #
###############################

C = Classifier()

C.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C.to(device)
C_optimizer = optim.SGD(C.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

criterion = nn.CrossEntropyLoss()

################################################
# sample selection and Batch training function # 
################################################
from train import update_Classifier as run_batch
###############################
# continual learning training #
###############################
ls_a = []


scaler = StandardScaler()

for task in range(config.nb_task):

  config.n_class = config.init_classes + task * config.n_inc
  config.task = task

  X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler = data_task_joint(config, X_train, Y_train, X_test, Y_test, scaler)
  config.nb_batch = int(len(X_train_t)/config.batchsize)

  if task > 0:
    C = C.expand_output_layer(config.init_classes, config.n_inc, task)
    C.to(device)

  for epoch in range(config.epochs):
    for n, (inputs, labels) in enumerate(train_loader):

      inputs = inputs.float()
      labels = labels.float()
      inputs = inputs.to(config.device)
      labels = labels.to(config.device)

      run_batch(config, C_optimizer, C, criterion, inputs, labels)
      print("\r", task, "task", epoch+1, "epoch", n, "/", config.nb_batch ,"batch", end="")
  print("\n")

  
  with torch.no_grad():
    accuracy = test(config, C, test_loader)
    ls_a.append(accuracy)
  print("task", task, "done")

print("joint", ", seed: ", config.seed_)
print("The Accuracy for each task:", ls_a)
print("The Global Average:", sum(ls_a)/len(ls_a))
