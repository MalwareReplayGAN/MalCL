import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from model import Classifier
from function import get_iter_test_dataset, test, oh, get_iter_train_dataset, get_dataloader
from data_ import get_ember_train_data, get_ember_test_data, shuffle_data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#######
# GPU #
#######

# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda")
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

data_dir = '/home/02mjpark/downloads/Continual_Learning_Malware_Datasets/EMBER_CL/EMBER_Class'
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

print("class_arr")
print(class_arr)

####################################################


print("class_arr")
print(class_arr)


##########
# Models #
##########

C = Classifier()
C.train()
C.to(device)

weight_decay = 0.000001
C_optimizer = optim.SGD(C.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

#############
# Functions # 
#############

def run_batch(C, C_optimizer, x_, y_):
    x_ = x_.view(-1, feats_length)
    x_ = Variable(x_)
    x_.to(device)

    C_optimizer.zero_grad()
    output = C(x_)
    output.to(device)

    C_loss = criterion(output, y_)

    C_loss.backward()
    C_optimizer.step()

    return output, C_loss


#######
# Run #
#######

scaler = StandardScaler()
ls_accuracy = []

for task in range(nb_task):

    n_class = init_classes + task * n_inc

    X_train_t, Y_train_t = get_iter_train_dataset(X_train, Y_train, n_class=n_class, n_inc=n_inc, task=task)
    train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=batchsize, n_class=n_class, scaler=scaler)

    X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class)

    if task>=0:
        C = C.expand_output_layer(init_classes, n_inc, task)
        C = C
        C.to(device)

    for epoch in range(epoch_number):
        
        train_loss = 0.0
        train_acc = 0.0

        for n, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device) #128,50
            labels = labels.to(device)

            outputs, loss = run_batch(C, C_optimizer, inputs, labels)
            train_loss += loss.item() * inputs.size(0) # calculate training loss and accuracy
            _, preds = torch.max(outputs, 1)
            class_label = torch.argmax(labels.data, dim=-1)
            train_acc += torch.sum(preds == class_label)
            print("\r", task, "task", epoch+1, "epoch", n, "batch", end="")

        print("epoch: ", epoch+1)
        train_loss = train_loss / len(X_train_t)
        train_acc = float(train_acc / len(X_train_t))
        print("train_loss: ", train_loss)
        print("train_acc: ", train_acc)

    with torch.no_grad():
        Accuracy = test(model=C, x_test=X_test_t, y_test=Y_test_t, n_class=n_class, device = device, scaler = scaler)
        ls_accuracy.append(Accuracy)

    print("task", task, "done")

    if task == nb_task-1:
        print("The Accuracy for each task:", ls_accuracy)
        print("The Global Average:", sum(ls_accuracy)/len(ls_accuracy))



            
