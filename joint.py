import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from model import Classifier
# from function2 import get_iter_test_dataset, get_dataloader, test, oh
from function2 import get_iter_test_dataset, test, oh, get_iter_train_dataset

from data_ import get_ember_train_data, get_ember_test_data
from sklearn.preprocessing import StandardScaler

#######
# GPU #
#######

# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"현재 사용 가능한 GPU의 수: {device_count}")
else:
    print("GPU를 사용할 수 없습니다.")

##############
# EMBER DATA #
##############

# Call the Ember Data

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_train, Y_train = get_ember_train_data(data_dir)
print("X_train", len(X_train))
X_test, Y_test, Y_test_onehot = get_ember_test_data(data_dir)

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
epoch_number = 100
z_dim = 62
momentum = 0.9
weight_decay = 0.000001

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

def get_dataloader(x, y, batchsize, n_class, scaler):

    # Manage Class Imbalance Issue
    y_ = np.array(y, dtype=int)
    class_sample_count = np.array([len(np.where(y_ == t)[0]) for t in np.unique(y_)])

    weight = 1. / class_sample_count

    samples_weight = np.array([weight[t%n_class] for t in y_])
    
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    x_ = torch.from_numpy(x).type(torch.FloatTensor)
    y_ = torch.from_numpy(y_).type(torch.FloatTensor)

    # Scaling
    #scaler = StandardScaler()
    scaler = scaler.fit(x_)
    x_ = scaler.transform(x_)
    x_ = torch.FloatTensor(x_)
    
    # One-hot Encoding
    y_oh = oh(y_, num_classes=n_class)
    y_oh = torch.Tensor(y_oh)

    data_tensored = torch.utils.data.TensorDataset(x_, y_oh)

    trainLoader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, num_workers=1, sampler=sampler)
    # trainLoader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize)

    return trainLoader, scaler


#######
# Run #
#######

scaler = StandardScaler()
ls_accuracy = []

for task in range(nb_task):

    n_class = init_classes + task * n_inc

    X_train_t, Y_train_t = get_iter_test_dataset(X_train, Y_train, n_class=n_class, n_inc=n_inc, task=task)
    train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=batchsize, n_class=n_class, scaler=scaler)

    X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class)

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
            inputs = inputs.to(device) #128,50
            labels = labels.to(device)

            outputs, loss = run_batch(C, C_optimizer, inputs, labels)
            train_loss += loss.item() * inputs.size(0) # calculate training loss and accuracy
            _, preds = torch.max(outputs, 1)
            class_label = torch.argmax(labels.data, dim=-1)
            train_acc += torch.sum(preds == class_label)

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





            
