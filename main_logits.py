import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
<<<<<<< HEAD:Main_Logits.py
from model import Generator, Discriminator, Classifier
from data_ import get_ember_train_data, extract_100data, oh, get_ember_test_data, shuffle_data
from function2 import get_iter_train_dataset, get_iter_test_dataset, selector, test, get_dataloader
=======
import joblib
from model import Generator, Discriminator
from data_ import get_ember_train_data, extract_100data, oh, get_ember_test_data, shuffle_data
from function import get_iter_train_dataset, get_iter_test_dataset, selector, test, get_dataloader
from torch.utils.data import TensorDataset
>>>>>>> 825518fc14f3cc3126dc9359f210ef62cb88d67b:main_logits.py
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import copy

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

#####################################
# Declarations and Hyper-parameters #
#####################################
seeds = [10]
init_classes = 50
final_classes = 100
n_inc = 5
nb_task = int(((final_classes - init_classes) / n_inc) + 1)
batchsize = 256
lr = 0.001
epoch_number = 50
z_dim = 62
ls_a = []
momentum = 0.9
weight_decay = 0.000001

##############
# EMBER DATA #
##############

# Call the Ember Data

data_dir = '/home/02mjpark/downloads/Continual_Learning_Malware_Datasets/EMBER_CL/EMBER_Class'
X_train, Y_train = get_ember_train_data(data_dir)
# X_train, Y_train = extract_100data(X_train, Y_train)
X_test, Y_test, Y_test_onehot = get_ember_test_data(data_dir)

feats_length= 2381
num_training_samples = 303331

############################################
# data random arange #
#############################################


import random
import copy
import matplotlib.pyplot as plt
'''
data_per_class = []

for i in range(final_classes):
  data_per_class.append(list(Y_train).count(i))

print("before random")

x = np.arange(final_classes)
plt.bar(x, data_per_class)

plt.show()
'''
#################################################

#class_arr = np.arange(final_classes)
class_arr_task_1 = np.arange(init_classes)
class_arr_next_tasks = np.arange(init_classes, final_classes)
indices = torch.randperm(50)
class_arr_next_tasks = torch.index_select(torch.Tensor(class_arr_next_tasks), dim=0, index=indices)
class_arr_next_tasks = np.array(class_arr_next_tasks)

class_arr = np.concatenate((class_arr_task_1, class_arr_next_tasks), axis = 0)
class_arr = list(class_arr)
Y_train_ = copy.deepcopy(Y_train)
Y_test_ = copy.deepcopy(Y_test)

for i in range(init_classes, final_classes):
  Y_train[np.where(Y_train_ == class_arr[i])] = i
  Y_test[np.where(Y_test_ == class_arr[i])] = i

####################################################


print("class_arr")
print(class_arr)




##########
# Models #
##########

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.output_dim = 5
        self.drop_prob = 0.5

        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_features, 1024, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 512, 3, 3, 1),
            nn.BatchNorm1d(512),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1)
        )
        
        self.fc1_f = nn.Flatten()
        self.fc1 = nn.Linear(128, self.output_dim)

        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        self.fc1_drop1 = nn.Dropout(self.drop_prob)
        self.fc1_act1 = nn.ReLU()
        
        self.softmax = nn.Softmax()

    def forward(self, x):

        # Get the original shape of the input tensor
        original_shape = x.size()

        # Reshape input data based on whether it's training or testing
        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)

        x = x.view(batch_size, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)

        x = self.fc1_f(x)
        x = self.fc1(x)
        x = self.fc1_bn1(x)
        x = self.fc1_drop1(x)
        x = self.fc1_act1(x)

        x = self.softmax(x)

        # If testing, reshape the output tensor back to the original shape
        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)

        return x
    

    def expand_output_layer(self, init_classes, nb_inc, task):
        """
        Expand the output layer to accommodate new_classes.
        This method retains the weights of the existing layer and expands it to fit the new class count.
        """
        # old_fc5 = self.fc5
        old_fc1 = self.fc1
        self.output_dim = init_classes + nb_inc * task

        # Create a new classifier layer with the updated output dimension.
        self.fc1 = nn.Linear(128, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)

        # Trnasfer the old weights and biases
        with torch.no_grad():
            self.fc1.weight[:old_fc1.out_features].copy_(old_fc1.weight.data)
            self.fc1.bias[:old_fc1.out_features].copy_(old_fc1.bias.data)

        return self

    def predict(self, x_data):
        result = self.forward(x_data)
        
        return result
    
    def get_logits(self, x):
        
        # Get the original shape of the input tensor
        original_shape = x.size()

        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)

        x = x.view(batch_size, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)

        logits = self.fc1_f(x)
        #logits = self.fc2(x)  # Get logits directly from the linear layer

        return logits

##############
# Parameters #
##############

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
C_optimizer = optim.SGD(C.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

k=2

#############
# Functions # 
#############

def run_batch(C, C_optimizer, x_, y_):

    x_ = x_.view([-1, feats_length])
    x_ = Variable(x_)
      
    # y_real and y_fake are the label for fake and true data

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
    if use_cuda:
       output = output.to(device)

    C_loss = criterion(output, y_)

    C_loss.backward()
    C_optimizer.step()

    return output, C_loss

<<<<<<< HEAD:Main_Logits.py
def get_replay_with_label(generator, classifier, batchsize, n_class, task):

    z_ = Variable(torch.rand((batchsize, z_dim)))
    if use_cuda:
        z_ = z_.to(device)

    images = generator(z_)
    logits = classifier.get_logits(images)

    if use_cuda:
        logits = logits.to(device)

    selected_samples = []

    loss_matrix = torch.zeros((batchsize, n_class)) 

    # Loop through each class to calculate cross-entropy loss
    for class_idx in range(n_class - 5*task):
        
        # Create a tensor of the current class index
        class_label = torch.full((batchsize,), class_idx, dtype=torch.long)
        if use_cuda:
            class_label = class_label.to(device)

        # Calculate the cross-entropy loss for the current class
        losses = criterion(logits, class_label)
        # Store the losses in the loss_matrix
        loss_matrix[:, class_idx] = losses
    
    # Initialize a list to store the best sample-class pairs
    selected_samples = [-1] * batchsize
    assigned_classes = set()

    # Iterate through each sample
    for sample_idx in range(batchsize):
        # Get the losses for the current sample across all classes
        sample_losses = loss_matrix[sample_idx]
        
        # Sort the losses and get the class with the lowest loss
        sorted_class_indices = torch.argsort(sample_losses)
        
        for class_idx in sorted_class_indices:
            class_idx = class_idx.item()
            if class_idx not in assigned_classes:
                selected_samples[sample_idx] = class_idx
                assigned_classes.add(class_idx)
                break
            
    # Ensure each class gets the best available sample
    final_selection = [[] for _ in range(n_class)]

    for class_idx in range(n_class):
        class_losses = []
        for sample_idx in range(batchsize):
            if selected_samples[sample_idx] == class_idx:
                class_losses.append((loss_matrix[sample_idx, class_idx], sample_idx))
        
        # Sort the class_losses and pick the best two(k) samples
        class_losses.sort()
        final_selection[class_idx] = [sample for _, sample in class_losses[:k]]
        
    selected_samples_indices = set(idx for class_samples in final_selection for idx in class_samples)
    selected_samples_indices = torch.LongTensor(list(selected_samples_indices))
    one_hot_selected_samples = torch.eye(n_class)[selected_samples_indices]
    selected_samples_tensor = torch.stack([images[idx] for idx in selected_samples_indices])

    return selected_samples_tensor.to(device), one_hot_selected_samples.to(device)
=======
def get_replay_with_label(generator, classifier, batchsize, n_class, task, logits_aver):
  arr = []
  z_ = Variable(torch.rand((batchsize, z_dim)))
  if use_cuda:
    z_ = z_.to(device)

  images = generator(z_)

  label = classifier.predict(images)
  logits = classifier.get_logits(images)
  if use_cuda:
     logits = logits.to(device)
>>>>>>> 825518fc14f3cc3126dc9359f210ef62cb88d67b:main_logits.py

  #print("logits.shape")
  #print(logits.shape)
    #L1 dist 구하
  for log in logits:
    arr.append(torch.mean(torch.abs(log - logits_aver)))
  '''
  print("log.shape")
  print(log.shape)
  print("(torch.abs(log - logits_aver)).shape")
  print((torch.abs(log - logits_aver)).shape)
  print("torch.mean(torch.abs(log - logits_aver)).shape")
  print(torch.mean(torch.abs(log - logits_aver)).shape)
  print("torch.mean(torch.abs(log - logits_aver))")
  print(torch.mean(torch.abs(log - logits_aver)))
  '''

  arr = torch.Tensor(arr).to(device)
  #print("arr.shape")
  #print(arr.shape)
  #arr = torch.array([arr, list(np.arange(len(arr)))])
  #arr = arr.transpose()
  #arr = arr.sort(key = lambda arr: arr[0])
  #for i in range(len(arr)):
  #  arr[i] = list(arr[i])
  #arr = np.array(arr)
  #arr.T
  if batchsize<(n_class-5)*2:
      sample_num = batchsize
  else: sample_num = (n_class-5)*2

  for_one_hot = torch.Tensor([list(i).index(max(i)) for i in label[arr.sort(0)[1][:sample_num]]])
  #print("for_one_hot", for_one_hot)
    #sample return
  return images[arr.sort(0)[1][:sample_num]].to(device), nn.functional.one_hot(for_one_hot.to(torch.int64), num_classes = n_class).to(device)


#########
# Train #
#########

G.reinit()
D.reinit()

new_f = open('duplicate', '+w')
new_f.write("")
new_f.close()

scaler = StandardScaler()

# Placeholder to store results for each task and each run
all_results = {task: [] for task in range(nb_task)}
logits_arr = []
logits_aver = 0

X_train, Y_train = shuffle_data(X_train, Y_train)

<<<<<<< HEAD:Main_Logits.py
    C = Classifier()
    C.train()
    C.to(device)

    weight_decay = 0.000001
    C_optimizer = optim.SGD(C.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for task in range(nb_task):
=======
for task in range(nb_task):
>>>>>>> 825518fc14f3cc3126dc9359f210ef62cb88d67b:main_logits.py
        
    new_f = open('duplicate', 'a')
    new_f.write(' '.join(['task', str(task), '\n']))
    new_f.close()

    n_class = init_classes + task * n_inc

        # Load data for the current task
    X_train_t, Y_train_t = get_iter_train_dataset(X_train,  Y_train, n_class=n_class, n_inc=n_inc, task=task)
    nb_batch = int(len(X_train_t)/batchsize)
    print("nb_batch", nb_batch)
    train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=batchsize, n_class=n_class, scaler = scaler)
    X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class=n_class)

    all_preds = []
    all_labels = []


    C = C.expand_output_layer(init_classes, n_inc, task)
    C = C
    C.to(device)

    metrics_per_epoch = [] # Placeholder for metrics per epoch
    for epoch in range(epoch_number):

        train_loss = 0.0
        train_acc = 0.0

        new_f = open('duplicate', 'a')
        new_f.write(' '.join(['task', str(task), '/ epoch', str(epoch), ': ']))
        new_f.close()

        for n, (inputs, labels) in enumerate(train_loader):
        
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            if task > 0 :
                    # We concat a batch of previously learned data.
                    # the more there are past tasks more data need to be regenerated.
                replay, re_label = get_replay_with_label(G_saved, C_saved, batchsize, n_class=n_class, task=task, logits_aver = logits_aver)
                inputs=torch.cat((inputs,replay),0)
                labels=torch.cat((labels,re_label),0)
            outputs, loss = run_batch(C,C_optimizer, inputs, labels)

            train_loss += loss.item() * inputs.size(0) # calculate training loss and accuracy
            _, preds = torch.max(outputs, 1)
            class_label = torch.argmax(labels.data, dim=-1)
            train_acc += torch.sum(preds == class_label)

                # Collecting all predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(class_label.cpu().numpy())
            if epoch == epoch_number-1:
                logits_arr.append(C.get_logits(inputs))
            print("\r", task, "task", epoch+1, "epoch", n, "batch", end="")
            
        new_f = open('duplicate', 'a')
        new_f.write('\n')
        new_f.close()
            
        print("epoch:", epoch+1)
        train_loss = train_loss / len(X_train_t)
        train_acc = float(train_acc / len(X_train_t))
        print("train_loss: ", train_loss)
        print("train_acc: ", train_acc)
        metrics_per_epoch.append(train_acc)
            

    G_saved = deepcopy(G)
    C_saved = deepcopy(C)
    all_results[task].append(metrics_per_epoch)
    #print("logits_arr.shape")
    #print(torch.stack(logits_arr).shape)
    logits_aver = torch.mean(torch.stack(logits_arr, dim=0), dim=0)
    logits_aver = torch.mean(logits_aver, dim = 0)
    #print("logits_aver.shape")
    #print(logits_aver.shape)

    #logits_arr = copy.deepcopy(torch.Tensor(logits_arr_new))
    #logits_arr = list(logits_arr)
    #print("logits_arr")
    #print(logits_arr)
    #print("logits_arr_new")
    #print(logits_arr_new)
    logits_arr = []

    ########
    # Test #
    ########

<<<<<<< HEAD:Main_Logits.py
        with torch.no_grad():
                accuracy = test(model=C, x_test=X_test_t, y_test=Y_test_t, n_class=n_class, device = device, scaler = scaler, seed=seed)
                ls_a.append(accuracy)

        print("task", task, "done")

        if task == nb_task-1:
            print("The Accuracy for each task:", ls_a)
            print("The Global Average:", sum(ls_a)/len(ls_a))

#########
# Plots #
#########
=======
    with torch.no_grad():
            
        accuracy = test(model=C, x_train = X_train_t, y_train = Y_train_t, x_test=X_test_t, y_test=Y_test_t, n_class=n_class, device = device, scaler = scaler)
        ls_a.append(accuracy)
>>>>>>> 825518fc14f3cc3126dc9359f210ef62cb88d67b:main_logits.py

    print("task", task, "done")

    if task == nb_task-1:
        print("The Accuracy for each task:", ls_a)
        print("The Global Average:", sum(ls_a)/len(ls_a))

<<<<<<< HEAD:Main_Logits.py
for task in range(nb_task):
    task_results = all_results[task]
    all_metrics = np.array(task_results) # Shape: (num_experiments, n_epochs)

    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)

    # Plot error bars
    plt.figure(figsize=(10, 6))
    epochs = range(1, epoch_number+1)
    plt.errorbar(epochs, mean_metrics, yerr=std_metrics, fmt='-o', capsize=5, label='Accuracy')
    plt.title(f'Error Bars for Task {task}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'error_bars_task_{task}.png'))
    plt.show()
    plt.close()

# PATH = " 모델 저장할 경로 .pt"    # 모델 저장할 경로로 수정
# torch.save(C.state_dict(), PATH)
# joblib.dump(scaler, ' scaler 저장 경로 .pkl')   # scaler 저장할 경로로 수정
=======
>>>>>>> 825518fc14f3cc3126dc9359f210ef62cb88d67b:main_logits.py