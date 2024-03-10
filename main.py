import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
from model import Generator, Discriminator, Classifier
from data_dir import get_ember_data
from data_ import extract_100data
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

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_train, Y_train, X_test, Y_test = get_ember_data(data_dir)
X_train_100, Y_train_100 = extract_100data(X_train, Y_train)
feats_length= 2381
num_training_samples = 303331

# Declarations and Hyper-parameters

init_classes = 50
final_classes = 100
nb_inc = 25
nb_task = int(((final_classes - init_classes) / nb_inc) + 1)
batchsize=16
lr = 0.001
epoch_number = 5
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

def get_replay_with_label(generator, classifier, batchsize, task, nb_inc):
    images_list = []
    labels_list = []
    task_label = [[] for _ in range(init_classes + (task-1) * nb_inc)]

    while True:
        if all(len(r) >= batchsize for r in task_label):
        # Checks whether there are at least 'batchsize' samples for each label in 'task_label'
        # The variable 'r' represents each innter list in 'task_label'
        # 'r' is a reference to one of the inner lists in 'task_label'
        # The loop continues until the condition is met for all inner lists, ensuring that each label has at least 'batchsize' samples.
            break
        z_ = Variable(torch.rand((batchsize, z_dim)))

        if use_cuda:
            z_ = z_.cuda(0)

        images = generator(z_)
        labels = classifier.predict(images)
        print(labels)
        
        for i in range(len(labels)):
            label = labels[i]
            # print(label)
            if (label < (init_classes + (task-1) * nb_inc)) and (len(task_label[label]) < batchsize):
                images_list.append(images[i].unsqueeze(0))
                labels_list.append(label.item())
                task_label[label].append(label.item())

        for i in range(len(task_label)):
            print("task_label:", i, "-", len(task_label[i]))
                

    images = torch.cat(images_list, dim=0)
    labels = torch.tensor(labels_list)

    return images.cpu(), labels.cpu()




# We reinit D and G to not cheat
G.reinit()
D.reinit()

for task in range(nb_task):
  # Load data for the current task
  x_, y_ = get_iter_dataset(X_train_100, Y_train_100, task=task, nb_inc=nb_inc)
  nb_batch = int(len(x_)/batchsize)
  # print("nb_batch", nb_batch)
  for epoch in range(epoch_number):
    for index in range(nb_batch):
      # print("index", index)
      current_x = torch.FloatTensor(x_[index*batchsize:(index+1)*batchsize])
      # print(current_x.shape)
      current_y = torch.LongTensor(y_[index*batchsize:(index+1)*batchsize])
      # print(current_y.shape)
      
      # print("y_ shape", y_.shape)

      if task > 0 :
        # We concat a batch of previously learned data
        # the more there are past tasks more data need to be regenerated
        replay, re_label = get_replay_with_label(G_saved, C_saved, batchsize, task, nb_inc)
        print(current_x.shape, replay.shape, re_label.shape)
        current_x=torch.cat((current_x,replay),0)
        current_y=torch.cat((current_y,re_label),0)

      run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, current_x, current_y)
    print("epoch:", epoch)

  G_saved = deepcopy(G)
  C_saved = deepcopy(C)

  print("task", task, "done")
  # z_ = Variable(torch.rand((nb_samples, z_dim)))

    
def test(model, x_test, y_test):

    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)


    if use_cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()

    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print('Accuracy: {}% Cost: {:.6f}'.format(
        correct_count / len(y_test) * 100, cost.item()
    ))    

test(C, X_test, Y_test)


