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
from function2 import get_iter_train_dataset, get_iter_test_dataset, selector, test, get_dataloader
import math
import time
from torch.utils.data import TensorDataset

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
ls_a = []
momentum = 0.9
weight_decay = 0.000001

##########
# Models #
##########

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.output_dim = 50
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

        x = self.fc1_f(x)
        logits = self.fc1(x)  # Get logits directly from the linear layer

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
      
      if use_cuda:
        x_, y_ = x_.to(device), y_.to(device)

      # update C

      C_optimizer.zero_grad()
      output = C(x_)
      if use_cuda:
         output = output.to(device)

      C_loss = criterion(output, y_)

      C_loss.backward()
      C_optimizer.step()

      return output, C_loss

def ground(a):
    new = np.zeros((a, a))
    for i in range(a):
        new[i][i] = 1
    return torch.Tensor(new)

def GetDist(logits, y2):
    sumArr = []
    for i in range(len(logits)):
        arr = []
        for (a, b) in zip(logits[i].tolist(), y2.tolist()):
            distances = criterion(logits, y2)
            arr.append(distances)
    return sumArr

def duplicate_index(index, c):
    for i in index:
        if c in i:
            return True
    return False

def duplicate(index):
    count = 0
    #print("index", index)
    for i in range(len(index)):
        for j in range(i+1, len(index)):
              count+=len(set(index[i]).intersection(set(index[j])))
    return count

def Rank(sumArr, img, y1, k, index_):

    img_ = []
    y1_ = []
    id = []

    index = [i for i in range(len(y1))]
    img_list = img.tolist()
    y1_list = y1.tolist()
#    zip(img_list, y1_list)

    y = pandas.DataFrame({'a': sumArr, 'b':img_list, 'c':y1_list, 'd':index})

    y = y.sort_values(by=['a'], axis = 0)

    for i in range(len(y['b'])):
        if len(id) == k:
            break
        if duplicate_index(index_, y['d'][i]):
            continue
        
        img_.append(y['b'][i])
        y1_.append(y['c'][i])
        id.append(y['d'][i])

    return img_, y1_, id

def selector(images, label, logits, k):
    img = []
    lbl = []
    lbl_for_one_hot = []
    index = []
    GroundTruth = ground(len(label[0]))
    #duplicate를 저장할 파일 열기

    new_f = open('duplicate', 'a')
    for i in range(len(GroundTruth)):
        sumArr = GetDist(logits, GroundTruth[i])
        new_images, new_label, new_index = Rank(sumArr, images, label, k, index)
        img = img + new_images
        lbl = lbl + new_label
        index = index + [new_index]
    duplicate_count = duplicate(index)
    #print(duplicate_count, end=" ")
    #파일에 작성
    new_f.write(str(duplicate_count))
    new_f.write('\t')
    new_f.close()
    for k in lbl:
      lbl_for_one_hot.append(k.index(max(k)))
    return torch.tensor(img), torch.tensor(lbl_for_one_hot)

def get_replay_with_label(generator, classifier, batchsize, n_class):

  z_ = Variable(torch.rand((batchsize, z_dim)))
  if use_cuda:
    z_ = z_.to(device)
  images = generator(z_)
  label = classifier.predict(images)
  logits = classifier.get_logits(images)
  images, lbl_for_one_hot = selector(images, label, logits, k)		#추가
  label = nn.functional.one_hot(lbl_for_one_hot, num_classes = len(label[0]))   #one hot encoding
  ex_lab = torch.Tensor(len(label)*[(n_class-len(label[0]))*[0]])
  label = torch.cat((label, ex_lab), 1)

  return images.to(device), label.to(device)



#########
# Train #
#########

G.reinit()
D.reinit()

new_f = open('duplicate', '+w')
new_f.write("")
new_f.close()

scaler = StandardScaler()

for task in range(nb_task):
  
  new_f = open('duplicate', 'a')
  new_f.write(' '.join(['task', str(task), '\n']))
  new_f.close()

  n_class = init_classes + task * n_inc

  # Load data for the current task
  X_train_t, Y_train_t = get_iter_train_dataset(X_train,  Y_train, n_class=n_class, n_inc=n_inc, task=task)
  nb_batch = int(len(X_train_t)/batchsize)
  train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=batchsize, n_class=n_class, scaler = scaler)
  X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class=n_class)

  if task > 0:
    C = C.expand_output_layer(init_classes, n_inc, task)
    C = C
    C.to(device)

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
          replay, re_label = get_replay_with_label(G_saved, C_saved, batchsize, n_class=n_class)
         #print("len(labels)", len(labels[0]))
         #print("len(re_label)", len(re_label[0]))
          inputs=torch.cat((inputs,replay),0)
          labels=torch.cat((labels,re_label),0)
      outputs, loss = run_batch(C,C_optimizer, inputs, labels)

      train_loss += loss.item() * inputs.size(0) # calculate training loss and accuracy
      _, preds = torch.max(outputs, 1)
      class_label = torch.argmax(labels.data, dim=-1)
      train_acc += torch.sum(preds == class_label)

      nb_per_10 = int(nb_batch/10)
      
    new_f = open('duplicate', 'a')
    new_f.write('\n')
    new_f.close()

    print("epoch:", epoch+1)
    train_loss = train_loss / len(X_train_t)
    train_acc = float(train_acc / len(X_train_t))
    print("train_loss: ", train_loss)
    print("train_acc: ", train_acc)

  G_saved = deepcopy(G)
  C_saved = deepcopy(C)
########
# Test #
########
    
  with torch.no_grad():
      accuracy = test(model=C, x_test=X_test_t, y_test=Y_test_t, n_class=n_class, device = device, scaler = scaler)
      ls_a.append(accuracy)

  print("task", task, "done")

  if task == nb_task-1:
     print("The Accuracy for each task:", ls_a)
     print("The Global Average:", sum(ls_a)/len(ls_a))



# PATH = " 모델 저장할 경로 .pt"    # 모델 저장할 경로로 수정
# torch.save(C.state_dict(), PATH)
# joblib.dump(scaler, ' scaler 저장 경로 .pkl')   # scaler 저장할 경로로 수정
