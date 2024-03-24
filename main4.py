import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
from model import Generator, Discriminator, Classifier
from data_ import get_ember_train_data, extract_100data
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


# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");
torch.manual_seed(0);


# Call the Ember Data

data_dir = '/home/gong77/EMBER_CL/EMBER_Class'
X_train, Y_train, Y_train_oh = get_ember_train_data(data_dir)
X_train_100, Y_train_100, Y_train_100_oh = extract_100data(X_train, Y_train)
feats_length= 2381
num_training_samples = 303331

# Declarations and Hyper-parameters

init_classes = 50
final_classes = 100
nb_inc = 25
nb_task = int(((final_classes - init_classes) / nb_inc) + 1)
batchsize = 64
lr = 0.001
epoch_number = 500
z_dim = 62

# k 추가
k = 16


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
        y_real_, y_fake_ = y_real_.cuda(0), y_fake_.cuda(0)

      z_ = torch.rand((x_.size(0), z_dim))

      x_, z_ = Variable(x_), Variable(z_)

      if use_cuda:
        x_, z_, y_ = x_.cuda(0), z_.cuda(0), y_.cuda(0)

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
         output = output.cuda(0)

      C_loss = criterion(output, y_)

      C_loss.backward()
      C_optimizer.step()

      return output

# def get_replay_with_label(generator, classifier, batchsize, task, nb_inc):
#     images_list = []
#     labels_list = []
#     task_label = [[] for _ in range(init_classes + (task-1) * nb_inc)]

#     while True:
#         if all(len(r) >= batchsize for r in task_label):
#         # Checks whether there are at least 'batchsize' samples for each label in 'task_label'
#         # The variable 'r' represents each innter list in 'task_label'
#         # 'r' is a reference to one of the inner lists in 'task_label'
#         # The loop continues until the condition is met for all inner lists, ensuring that each label has at least 'batchsize' samples.
#             break
#         z_ = Variable(torch.rand((batchsize, z_dim)))

#         if use_cuda:
#             z_ = z_.cuda(0)

#         images = generator(z_)
#         labels = classifier.predict(images)
        
#         for i in range(len(labels)):
#             label = labels[i]
#             # print(label)
#             if (label < (init_classes + (task-1) * nb_inc)) and (len(task_label[label]) < batchsize):
#                 images_list.append(images[i].unsqueeze(0))
#                 labels_list.append(label.item())
#                 task_label[label].append(label.item())

#         for i in range(len(task_label)):
#             print("task_label:", i, "-", len(task_label[i]))
                

#     images = torch.cat(images_list, dim=0)
#     labels = torch.tensor(labels_list)

#     return images.cpu(), labels.cpu()


#추가

# 유클리드 거리
def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y)**2)).item()

# ground truth
def g_t(classes):
    return torch.eye(classes)

# top k개의 index 반환
import heapq

def top_k(nums, k):
    min_heap = []
    for i, num in enumerate(nums):
        heapq.heappush(min_heap, (num, i))  # 튜플 형태로 요소와 인덱스 저장
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # 가장 작은 요소 제거
    return [index for _, index in min_heap]


def get_replay_with_label(generator, classifier, batchsize, task, nb_inc, k):
    images_list = []
    labels_list = []

    k_label = [[] for _ in range(init_classes + (task-1) * nb_inc)]

    z_ = Variable(torch.rand((k * (init_classes + (task-1) * nb_inc), z_dim)))

    if use_cuda:
        z_ = z_.cuda(0)

    images = generator(z_)
    labels = classifier.pro(images)

    gr = g_t(len(labels[0])) #ground truth

    if use_cuda:
        gr = gr.cuda(0)

    # labels 수에 맞춰 distance 리스트 길이 정하기
    distance_label = [[] for _ in range(len(labels[0]))]

    for i in range(len(labels[0])):
        for j in range(len(labels)):
            distance_label[i].append(euclidean_distance(labels[j], gr[i]))

    for i in range(init_classes + (task-1) * nb_inc):
        k_label[i] = top_k(distance_label[i], k)

    for i in range(init_classes + (task-1) * nb_inc):
        for j in range(k):
            images_list.append(images[k_label[i][j]].unsqueeze(0))
            labels_list.append(i)

    images = torch.cat(images_list, dim=0)
    labels = torch.tensor(labels_list)

    return images.cpu(), labels.cpu()



# We reinit D and G to not cheat
G.reinit()
D.reinit()

for task in range(nb_task):
  # Load data for the current task
  x_, y_ = get_iter_dataset(X_train_100, Y_train_100_oh, task=task, nb_inc=nb_inc)
  nb_batch = int(len(x_)/batchsize)
  # print("nb_batch", nb_batch)
  for epoch in range(epoch_number):
    for index in range(nb_batch):
      # print("index", index)
      x_i = torch.FloatTensor(x_[index*batchsize:(index+1)*batchsize])
      # print(x_i.shape)
      y_i = torch.Tensor(y_[index*batchsize:(index+1)*batchsize])
      y_i = y_i.float()
      # print(y_i.shape)
      
      # print("y_ shape", y_.shape)

      if task > 0 :
        # We concat a batch of previously learned data
        # the more there are past tasks more data need to be regenerated
        replay, re_label = get_replay_with_label(G_saved, C_saved, batchsize, task, nb_inc, k)
        # print(x_i.shape, replay.shape, re_label.shape)
        x_i=torch.cat((x_i,replay),0)
        y_i=torch.cat((y_i,re_label),0)

      run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, x_i, y_i)
    print("epoch:", epoch)

  G_saved = deepcopy(G)
  C_saved = deepcopy(C)

  print("task", task, "done")
  # z_ = Variable(torch.rand((nb_samples, z_dim)))



PATH = "/home/gong77/ConvGAN/SAVE/mdl_100.pt"
torch.save(C.state_dict(), PATH)

