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
from function import get_iter_train_dataset, get_iter_test_dataset, selector, test, get_dataloader
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
torch.manual_seed(10)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"현재 사용 가능한 GPU의 수: {device_count}")
    device = torch.device("cuda")
else:
    print("GPU를 사용할 수 없습니다.")

##############
# EMBER DATA #
##############

# Call the Ember Data

data_dir = '/home/02mjpark/downloads/Continual_Learning_Malware_Datasets/EMBER_CL/EMBER_Class'
X_train, Y_train = get_ember_train_data(data_dir)
#X_train, Y_train = extract_100data(X_train, Y_train)
#print("X_train", len(X_train))
X_test, Y_test, Y_test_onehot = get_ember_test_data(data_dir)

# X_train_100, Y_train_100 = extract_100data(X_train, Y_train)
# Y_train_oh = oh(Y_train)
# Y_train_100_oh = oh(Y_train_100)
feats_length= 2381
num_training_samples = 303331

#####################################
# Declarations and Hyper-parameters #
#####################################

init_classes = 50
final_classes = 100
n_inc = 5
nb_task = int(((final_classes - init_classes) / n_inc) + 1)
batchsize = 128
lr = 0.001
epoch_number = 10
z_dim = 62
k = 2

############################################
# data random arange #
#############################################


import random
import copy
import matplotlib.pyplot as plt

data_per_class = []

for i in range(final_classes):
  data_per_class.append(list(Y_train).count(i))

print("before random")

x = np.arange(final_classes)
plt.bar(x, data_per_class)

plt.show()


#class_arr = np.arange(final_classes)
class_arr_task_1 = np.arange(init_classes)
class_arr_next_tasks = np.arange(init_classes, final_classes)
random.shuffle(class_arr_next_tasks)
class_arr = np.concatenate((class_arr_task_1, class_arr_next_tasks), axis = 0)
class_arr = list(class_arr)
Y_train_ = copy.deepcopy(Y_train)
Y_test_ = copy.deepcopy(Y_test)

for i in range(init_classes, final_classes):
  Y_train[np.where(Y_train_ == class_arr[i])] = i
  Y_test[np.where(Y_test_ == class_arr[i])] = i

print("class_arr")
print(class_arr)




data_per_class_after = []

for i in range(final_classes):
  data_per_class_after.append(list(Y_train).count(i))

print("after random")

x = np.arange(final_classes)
plt.bar(x, data_per_class_after)

plt.show()


for i in range(final_classes):
  if data_per_class[class_arr[i]] != data_per_class_after[i]:
     print("the amount of data is not same as before random change")




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

'''
if use_cuda:
    G.cuda(0)
    D.cuda(0)
    C.cuda(0)
'''
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
C_optimizer = optim.Adam(C.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

#############
# Functions # 
#############

def run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, x_, y_):
      x_ = x_.view([-1, feats_length])
      # print("x_ shape", x_.shape) # [batchsize, feats_length] 16, 2381

      # y_real and y_fake are the label for fake and true data
      y_real_ = Variable(torch.ones(x_.size(0), 1))
      y_fake_ = Variable(torch.zeros(x_.size(0), 1))
      # print("y_real_shape", y_real_.shape) # [batchsize, 1] 16, 1

      if use_cuda:
        y_real_, y_fake_ = y_real_.to(device), y_fake_.to(device)

      z_ = torch.rand((x_.size(0), z_dim))

      x_, z_ = Variable(x_), Variable(z_)

      if use_cuda:
        x_, z_, y_ = x_.to(device), z_.to(device), y_.to(device)

      # update D network
      D_optimizer.zero_grad()

      D_real = D(x_)
      # print("D_real shape", D_real.shape) # [16, 1]
      # print("y_real_[:x_.size(0)].shape: ", y_real_[:x_.size(0)].shape) # [16, 1]
      D_real_loss = BCELoss(D_real, y_real_[:x_.size(0)])

      G_ = G(z_)
      # print('G_ shape', G_.shape) # 16, 2381
      D_fake = D(G_)
      # print("D_fake shape", D_fake.shape) # 16, 1
      # print("y_fake_[:x_.size(0)] shape", y_fake_[:x_.size(0)].shape) # 16, 1
      D_fake_loss = BCELoss(D_fake, y_fake_[:x_.size(0)])

      D_loss = D_real_loss + D_fake_loss

      D_loss.backward()
      D_optimizer.step()

      # update G network
      G_optimizer.zero_grad()

      G_ = G(z_)
      D_fake = D(G_)
      G_loss = BCELoss(D_fake, y_real_[:x_.size(0)])

      G_loss.backward()
      G_optimizer.step()

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


def get_replay_with_label(generator, classifier, batchsize, n_class):

  z_ = Variable(torch.rand((batchsize, z_dim)))
  if use_cuda:
    z_ = z_.to(device)
  images = generator(z_)
  label = classifier.predict(images)
  #print("len(images)", len(images[0]))
  #print("len(label)", len(label[0]))
#  print("type(images), type(label)", type(images), type(label))
  images, lbl_for_one_hot = selector(images, label, k)		#추가
  label = nn.functional.one_hot(lbl_for_one_hot, num_classes = len(label[0]))   #one hot encoding
  ex_lab = torch.Tensor(len(label)*[(n_class-len(label[0]))*[0]])
  label = torch.cat((label, ex_lab), 1)
  '''
  print("========================== generated images ===========================")
  print(images)
  print("++++++++++++++++++++++++++ labels ++++++++++++++++++++++++++++++++++")
  print(label)
  '''
  #print("num of label: ", len(label))
  return images.to(device), label.to(device)


def get_replay_with_label_no_select(generator, classifier, batchsize, n_class):

  for_one_hot = []
  z_ = Variable(torch.rand((batchsize, z_dim)))
  if use_cuda:
    z_ = z_.to(device)
  images = generator(z_)
  label = classifier.predict(images)
  for k in label:
    k = list(k)
    for_one_hot.append(k.index(max(k)))
  label = nn.functional.one_hot(torch.tensor(for_one_hot), num_classes = len(label[0]))   #one hot encoding

  #print("len(images)", len(images[0]))
  #print("len(label)", len(label[0]))
#  print("type(images), type(label)", type(images), type(label))
#  images, lbl_for_one_hot = selector(images, label, k)		#추가
#  label = nn.functional.one_hot(lbl_for_one_hot, num_classes = len(label[0]))   #one hot encoding
  ex_lab = torch.Tensor(len(label)*[(n_class-len(label[0]))*[0]])
  label = torch.cat((label, ex_lab), 1)
  '''
  print("========================== generated images ===========================")
  print(images)
  print("++++++++++++++++++++++++++ labels ++++++++++++++++++++++++++++++++++")
  print(label)
  '''
  #print("num of label: ", len(label))
  return images.to(device), label.to(device)




# We reinit D and G to not cheat
G.reinit()
D.reinit()

import datetime
result_name = datetime.datetime.now()
date_str = result_name.strftime("%Y.%m.%d.%H:%M:%S")
print(date_str)
result_name = "result_" + date_str
result_f = open(result_name, '+w')
result_f.write("")
result_f.close()

test_accuracy_array = []
new_f = open('duplicate', '+w')
new_f.write("")
new_f.close()

scaler = StandardScaler()

for task in range(nb_task):
  new_f = open('duplicate', 'a')
  new_f.write(' '.join(['task', str(task), '\n']))
  new_f.close()

  result_f = open(result_name, 'a')
  result_f.write(' '.join(['task', str(task), '\n']))
  result_f.close()

  n_class = init_classes + task * n_inc
  # Load data for the current task
  print("get_iter_train_dataset")
  X_train_t, Y_train_t = get_iter_train_dataset(X_train,  Y_train, n_class=n_class, n_inc=n_inc, task=task)
  nb_batch = int(len(X_train_t)/batchsize)
  print("nb_batch", nb_batch)
  print("len(X_train_t)", len(X_train_t))
  print("len(X_train_t[0])", len(X_train_t[0]))
  print("get_dataloader")
  train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=batchsize, n_class=n_class, scaler = scaler)
  print("get_iter_test_dataset")
  X_test, Y_test = get_iter_test_dataset(X_test, Y_test, n_class=n_class)

  for epoch in range(epoch_number):
    train_loss = 0.0
    train_acc = 0.0
    new_f = open('duplicate', 'a')
    new_f.write(' '.join(['task', str(task), '/ epoch', str(epoch), ': ']))
    new_f.close()

#    result_f = open(result_name, 'a')
#    result_f.write(' '.join(['task', str(task), '/ epoch', str(epoch), ': ']))
#    result_f.close()

    for n, (inputs, labels) in enumerate(train_loader):
      
      inputs = inputs.float()
      labels = labels.float()
      inputs = inputs.to(device)
      labels = labels.to(device)
        
      if task > 0 :
        # We concat a batch of previously learned data.
        # the more there are past tasks more data need to be regenerated.
          replay, re_label = get_replay_with_label(G_saved, C_saved, batchsize, n_class=n_class)
          #print("len(labels)", len(labels[0]))
          #print("len(re_label)", len(re_label[0]))
          inputs=torch.cat((inputs,replay),0)
          labels=torch.cat((labels,re_label),0) 

      C = C.expand_output_layer(init_classes, n_inc, task)
      C = C
      C.to(device)
#        print("before run_batch n", n)
      outputs, loss = run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, inputs, labels)
#        print("after run_batch n", n)
      train_loss += loss.item() * inputs.size(0) # calculate training loss and accuracy
      _, preds = torch.max(outputs, 1)
#        print("preds", preds)
#        print("labels.data", labels.data)
      class_label = torch.argmax(labels.data, dim=-1)
#        print("class_label", class_label)
#        print("torch.sum(preds == labels.data)", torch.sum(preds == class_label))
      train_acc += torch.sum(preds == class_label)

      nb_per_10 = int(nb_batch/10)
      print("\r", task, "task", epoch+1, "epoch", n, "batch", end="")
      #if (n+1)%nb_per_10 == 0:
#          print("orejgia")
          #print("\r", "> "*int((n+1)/nb_per_10), "[", (n+1), "/", nb_batch, "]", end="")
    print("\n")
    new_f = open('duplicate', 'a')
    new_f.write('\n')
    new_f.close()

    print("epoch:", epoch+1)
    train_loss = train_loss / len(X_train_t)
    train_acc = float(train_acc / len(X_train_t))
    print("train_loss: ", train_loss)
    print("train_acc: ", train_acc)

    train_result = "epoch: " + str(epoch+1) + "\ntrain_loss: " + str(train_loss) + "\ntrain_acc: " + str(train_acc) + "\n"
    result_f = open(result_name, 'a')
    result_f.write(train_result)
    result_f.close()

  G_saved = deepcopy(G)
  C_saved = deepcopy(C)
  #if task == 0:
  #  date_time = datetime.datetime.now()
  #  date_time = date_time.strftime("%Y.%m.%d.%H:%M:%S")
  #  C_PATH_0 = "C_model_task_0"+date_time+".pt"    # 모델 저장할 경로로 수정
  #  torch.save(C.state_dict(), C_PATH_0)
  #  G_PATH_0 = "G_model_task_0"+date_time+".pt"    # 모델 저장할 경로로 수정
  #  torch.save(C.state_dict(), C_PATH_0)

  # test
    
  with torch.no_grad():
      ls_accuracy = test(model=C_saved, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, n_class=n_class, device = device, scaler = scaler)
      test_accuracy_array.append(ls_accuracy)
      test_result = "test_acc: " + str(ls_accuracy) + "\n"
      result_f = open(result_name, 'a')
      result_f.write(test_result)
      result_f.close()
  print("task", task, "done")

  if task == nb_task-1:
     print("The Accuracy for each task:", test_accuracy_array)
     print("The Global Average:", sum(test_accuracy_array)/len(test_accuracy_array))



PATH = " 모델 저장할 경로 .pt"    # 모델 저장할 경로로 수정
torch.save(C.state_dict(), PATH)
joblib.dump(scaler, ' scaler 저장 경로 .pkl')   # scaler 저장할 경로로 수정