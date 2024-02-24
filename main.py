import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
from model import Generator, Discriminator, Classifier
from data_dir import get_ember_data
import matplotlib.pyplot as plt

# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");
torch.manual_seed(0);


parser = argparse.ArgumentParser()

# Call the Ember Data

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_train, Y_train, X_test, Y_test = get_ember_data(data_dir)

feats_length= 2381
num_training_samples = 303331

# Function

def get_iter_dataset(x_train, y_train, task=None):
   if task is not None:
    if task == 0:
      return x_train[np.where(y_train < 50)[0]]
    else:
      start = 50 + (task-1) * 5
      end = 50 + task * 5
      return x_train[np.where((y_train >= start) & (y_train < end))]
   
def run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, x_, t_):
      x_ = x_.view([-1,feats_length]))

      # y_real and y_fake are the label for fake and true data
      y_real_ = Variable(torch.ones(x_.size(0), 1))
      y_fake_ = Variable(torch.zeros(x_.size(0), 1))

      if use_cuda:
        y_real_, y_fake_ = y_real_.cuda(0), y_fake_.cuda(0)

      z_ = torch.rand((x_.size(0), z_dim))

      x_, z_ = Variable(x_), Variable(z_)

      if use_cuda:
        x_, z_, t_ = x_.cuda(0), z_.cuda(0), t_.cuda(0)

      # update D network
      D_optimizer.zero_grad()

      D_real = D(x_)
      D_real_loss = BCELoss(D_real, y_real_[:x_.size(0)])

      G_ = G(z_)
      D_fake = D(G_)
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

      output = C(x_)

      C_loss = criterion(output, t_)

      C_loss.backward()
      C_optimizer.step()

      return output

def get_replay_with_label(generator, classifier, batchsize, task):
    # calculate the number of data to generate in each class

    # define the lists for the generated data and labels.
    gen_data_list = []
    labels_list = []
    # generate the data and allocate the label in each class in the loop
    for class_idx in range(task):
        # calculate the number of data to generate in each class
        num_data_to_generate = batchsize
        # 생성된 이미지 개수가 num_data_to_generate가 될 때까지 반복하여 이미지를 생성하고 라벨을 할당합니다.
        while num_data_to_generate > 0:
            # generate the new data
            if num_data_to_generate == 1:
              z_ = Variable(torch.rand((min(2, batchsize), z_dim)))
              if use_cuda:
                  z_ = z_.cuda(0)
              gen_data = generator(z_)
              label = classifier.predict(gen_data)
              class_data = label for label in gen_data if ((label < 50+5*task) & (label >= 50+5*(task-1)))
              class_labels = label for label in label if ((label < 50+5*task) & (label >= 50+5*(task-1)))
              if len(class_data) > 0:
                gen_data_list.append(class_data[0:1])
                labels_list.append(class_labels[0:1])
                num_data_to_generate -= class_data.size(0)
            else:
              z_ = Variable(torch.rand((min(num_data_to_generate, batchsize), z_dim)))
              if use_cuda:
                  z_ = z_.cuda(0)
              gen_data = generator(z_)
              # 생성된 이미지를 분류기로 분류하여 라벨을 생성합니다.
              label = classifier.predict(gen_data)
              # 생성된 이미지 중에서 현재 클래스에 해당하는 이미지만 선택하여 저장합니다.
              if task > 0:
                class_data = label for label in gen_data if ((label < 50+5*task) & (label >= 50+5*(task-1)))
                class_labels = label for label in label if ((label < 50+5*task) & (label >= 50+5*(task-1)))
              else:
                class_data = gen_data[label < 50]
                class_labels = label[label < 50]
              # 선택된 이미지와 라벨을 리스트에 추가합니다.
              gen_data_list.append(class_data)
              labels_list.append(class_labels)
              # update the number of generated data.
              num_data_to_generate -= class_data.size(0)

    # 생성된 이미지와 라벨을 하나로 합칩니다

    gen_data_s = torch.cat(gen_data_list, dim=0)


    labels = torch.cat(labels_list, dim=0)

    return gen_data_s.cpu(), labels.cpu()


# Declarations and Hyper-parameters

init_classes = 50
final_classes = 100
nb_task = 11
add_ = 5
batchsize=256
lr = 0.001
epoch_number = 10
z_dim = 62


G = Generator()
D = Discriminator()
C = Classifier()

G.train()
D.train()
C.train()

if use_cuda:
    G.cuda(0)
    D.cuda(0)
    C.cuda(0)

G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
C_optimizer = optim.Adam(C.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

#

# We reinit D and G to not cheat
G.reinit()
D.reinit()

for task in range(nb_task):
  # Load data for the current task
  data = get_iter_dataset(X_train, Y_train, task)
  nb_batch = int(len(data)/batchsize)

  for epoch in range(epoch_number):
    for index in range(nb_batch):
      x_=torch.FloatTensor(data[index*batchsize:(index+1)*batchsize])
      t_ = torch.full((batchsize,), task, dtype=torch.long)

      if task > 0 :
        # We concat a batch of previously learned data
        # the more there is past task more data need to be regenerate
        replay, re_label = get_replay_with_label(G_saved, C_saved, batchsize, task)
        x_=torch.cat((x_,replay),0)
        t_=torch.cat((t_,re_label),0)

      run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, x_, t_)

  G_saved = deepcopy(G)
  C_saved = deepcopy(C)

  z_ = Variable(torch.rand((nb_samples, z_dim)))

    
    