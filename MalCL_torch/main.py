
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
import joblib
from models import Generator, Discriminator, Classifier
from function import test, class_pick_rand

from torch.utils.data import TensorDataset
from dynaconf import Dynaconf
from sample_selection import L2_One_Hot, L1_B_Mean, L1_C_Mean
from arguments import _parse_args
from setting import configurate, torch_setting
from _data import dataset
from train import data_task, report_result, mean_logits, collect_logits, col_arr
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

G_optimizer = optim.Adam(G.parameters(), lr=config.lr)
D_optimizer = optim.Adam(D.parameters(), lr=config.lr)
C_optimizer = optim.SGD(C.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

################################################
# sample selection and Batch training function # 
################################################

if config.sample_select == 'L2_One_Hot' : from sample_selection import common_vars, L2_One_Hot as get_replay_with_label
elif config.sample_select == 'L1_B_Mean' : from sample_selection import common_vars, L1_B_Mean as get_replay_with_label
elif config.sample_select == 'L1_C_Mean' : from sample_selection import common_vars, L1_C_Mean as get_replay_with_label

if config.Generator_loss == 'FML' : from train import run_batch_FML as run_batch
elif config.Generator_loss == 'BCE' : from train import run_batch_BCE as run_batch

###############################
# continual learning training #
###############################
ls_a = []
ls_a_old = []
G.reinit()
D.reinit()


scaler = StandardScaler()
x_ = torch.from_numpy(X_train).type(torch.FloatTensor)


###############################

print(f"before train")
for task in range(config.nb_task):

  config.n_class = config.init_classes + task * config.n_inc
  config.task = task

  X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler = data_task(config, X_train, Y_train, X_test, Y_test, scaler)
  config.nb_batch = int(len(X_train_t)/config.batchsize)

  logits_collect = col_arr(config, X_train_t)


  if task > 0:
    C = C.expand_output_layer(config.init_classes, config.n_inc, task)
    C.to(device)

  for epoch in range(config.epochs):
    for n, (inputs, labels) in enumerate(train_loader):

      inputs = inputs.float()
      labels = labels.float()
      inputs = inputs.to(config.device)
      labels = labels.to(config.device)

      if config.task > 0 :
        synthetic, pred_label, logits_gen = common_vars(config, past_Generator, past_Classifier)
        replay, re_label = get_replay_with_label(config, synthetic, pred_label, logits_gen, logits_real)
        inputs=torch.cat((inputs,replay),0)
        labels=torch.cat((labels,re_label),0) 
      C.train()
      G.train()
      D.train()
      run_batch(config, G, D, C, G_optimizer, D_optimizer, C_optimizer, criterion, BCELoss, inputs, labels)
      logits_collect = collect_logits(config, C, logits_collect, inputs, labels, n)
      print("\r", task, "task", epoch+1, "epoch", n, "/", config.nb_batch ,"batch", end="")
  print("\n")

  # past
  past_Generator = deepcopy(G)
  past_Classifier = deepcopy(C)

  logits_real = mean_logits(config, logits_collect)

  
  with torch.no_grad():
    print("test_new")
    accuracy = test(config, C, test_loader)
    ls_a.append(accuracy)
  print("task", task, "done")

report_result(config, ls_a)