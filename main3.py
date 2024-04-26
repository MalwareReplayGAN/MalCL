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
from data_ import get_ember_train_data, extract_100data, oh


# from function import get_iter_dataset, run_batch, get_replay_with_label

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


##########################


use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()
torch.manual_seed(0)

if torch.cuda.is_available():
   device_count = torch.cuda.device_count()
   print(f"The number of current usable GPU: {device_count}")
else:
   print("Cannot use GPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############
# EMBER DATA #
##############

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_train, Y_train = get_ember_train_data(data_dir)
X_train_100, Y_train_100 = extract_100data(X_train, Y_train)
# Y_train_oh = oh(Y_train)
# Y_train_100_oh = oh(Y_train_100)
feats_length= 2381
num_training_samples = 303331

#####################################
# Declarations and Hyper-parameters #
#####################################

init_classes = 20
final_classes = 100
nb_inc = 20
nb_task = int(((final_classes - init_classes) / nb_inc) + 1)
batchsize = 128
lr = 0.001
epoch_number = 100
z_dim = 62

scaler = StandardScaler()

##########
# Models #
##########

G = Generator()
D = Discriminator()
C = Classifier()

# if use_cuda:
#     G.to(device)
#     D.to(device)
#     C.to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
C_optimizer = optim.Adam(C.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

#

# def get_iter_dataset(x_train, y_train, y_train_oh, task, nb_inc=None):
#    if task is not None:
#     if task == 0:
#        selected_indices = np.where(y_train < init_classes)[0]
#        return x_train[selected_indices], y_train_oh[selected_indices]  
#     else:
#        start = init_classes + (task-1) * nb_inc
#        end = init_classes + task * nb_inc
#        selected_indices = np.where((y_train >= start) & (y_train < end))
#        return x_train[selected_indices], y_train_oh[selected_indices]
    

def get_iter_dataset(x_train, y_train, task, nb_inc=None):
   if task is not None:
    if task == 0:
       selected_indices = np.where(y_train < init_classes)[0]
       return x_train[selected_indices], y_train[selected_indices]  
    else:
       start = init_classes + (task-1) * nb_inc
       end = init_classes + task * nb_inc
       selected_indices = np.where((y_train >= start) & (y_train < end))
       return x_train[selected_indices], y_train[selected_indices]

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

      return output

def ground(a):
    new = np.zeros((a, a))
    for i in range(a):
        new[i][i] = 1
    return new

def Rank(sumArr, img, y1, k):
    img_list = img.tolist()
    y1_list = y1.tolist()
    zip(img_list, y1_list)
    y = pandas.DataFrame({'a': sumArr, 'b':img.tolist(), 'c':y1.tolist()})
    y = y.sort_values(by=['a'], axis = 0)
    img_ = y['b'][0:k]
    y1_ = y['c'][0:k]
    return img_.tolist(), y1_.tolist()

def GetL2Dist(y1, y2):
    sumArr = []
    for i in range(len(y1)):
        arr = []
        for (a, b) in zip(y1[i].tolist() ,y2.tolist()):
            arr.append((a-b)**2)
        sumArr.append(sum(arr))
    return sumArr

def GetCrossEntropy(y1, y2):
    sumArr = []
    delta = 1e-7 # log 0을 계산할 수 없어서 임의의 작은 값 넣어줌
    for i in range(len(y1)):
      arr = []
      for (a, b) in zip(y1[i].tolist() ,y2.tolist()):
          arr.append(-a*np.log(b+delta))
      sumArr.append(sum(arr))
    return sumArr

def selector(images, label, k):
    img = []
    lbl = []
    lbl_for_one_hot = []
    GroundTruth = ground(len(label[0]))
    for i in range(len(GroundTruth)):
        sumArr = GetL2Dist(label, GroundTruth[i])
        new_images, new_label = Rank(sumArr, images, label, k)
        img = img + new_images
        lbl = lbl + new_label
    # duplicate_count = duplicate(index)
    # print(duplicate_count, end=" ")
    for k in lbl:
      lbl_for_one_hot.append(k.index(max(k)))
    return torch.tensor(img), torch.tensor(lbl_for_one_hot)

def duplicate(index):
    count = 0
    for i in range(len(index)):
        for j in range(i+1, len(index)):
              count+=len(set(index[i]).intersection(set(index[j])))
    return count


#수정함
k = 1

def get_replay_with_label(generator, classifier, batchsize):

  z_ = Variable(torch.rand((batchsize, z_dim)))
  if use_cuda:
    z_ = z_.to(device)
  images = generator(z_)
  label = classifier.predict(images)
  images, lbl_for_one_hot = selector(images, label, k)		#추가
  label = nn.functional.one_hot(lbl_for_one_hot, num_classes = len(label[0]) + nb_inc)   #one hot encoding
  torch.tensor(label)
  return images.to(device), label.to(device)


# We reinit D and G to not cheat
G.reinit()
D.reinit()


# G = nn.DataParallel(G)
# D = nn.DataParallel(D)
# C = nn.DataParallel(C)

G.to(device)
D.to(device)
C.to(device)

G.train()
D.train()
C.train()



for task in range(nb_task):
  # Load data for the current task
  x_, y_ = get_iter_dataset(X_train, Y_train, task=task, nb_inc=nb_inc)
  y_oh = oh(y_, num_classes=init_classes+nb_inc*task)
  x_ = scaler.fit_transform(x_)
  nb_batch = int(len(x_)/batchsize)
  
  for epoch in range(epoch_number):
    for index in range(nb_batch):
      x_i = torch.FloatTensor(x_[index*batchsize:(index+1)*batchsize])
      y_i = torch.Tensor(y_oh[index*batchsize:(index+1)*batchsize])
      y_i = y_i.float()

      x_i = x_i.to(device)
      y_i = y_i.to(device)

      # print(x_i.shape) # 64, 2381
      # print(y_i.shape) # 64, 20/40/60/
      
      if task > 0 :
        # We concat a batch of previously learned data
        # the more there are past tasks more data need to be regenerated
        # C_saved = C_saved.expand_output_layer(new_classes=init_classes+nb_inc*task)
        
        replay, re_label = get_replay_with_label(G_saved, C_saved, batchsize)
        x_i=torch.cat((x_i,replay),0)
        y_i=torch.cat((y_i,re_label),0)

        C = C.expand_output_layer(init_classes, nb_inc, task)
        C = C
        C.to(device)

      run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, x_i, y_i)
    print("epoch:", epoch)

  G_saved = deepcopy(G)
  C_saved = deepcopy(C)

  print("task", task, "done")
  # z_ = Variable(torch.rand((nb_samples, z_dim)))



PATH = "/home/02mjpark/ConvGAN/SAVE/mdl.pt"
torch.save(C.state_dict(), PATH)
joblib.dump(scaler, '/home/02mjpark/ConvGAN/SAVE/scaler_main3.pkl')