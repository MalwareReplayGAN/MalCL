import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_ import get_ember_train_data, extract_100data, shuffle_data, oh
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib


# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu");
# torch.manual_seed(0);


##############
# EMBER DATA #
##############

data_dir = '/home/02mjpark/downloads/Continual_Learning_Malware_Datasets/EMBER_CL/EMBER_Class'
X_train, Y_train = get_ember_train_data(data_dir) # get the ember train data
Y_train_oh = oh(Y_train, num_classes=100) # one hot encoding of Y_train
X_train, Y_train_oh = shuffle_data(X_train, Y_train_oh, 0) 

####################
# Classifier Model #
####################

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.output_dim = 100
        self.drop_prob = 0.5
        self.batchsize = 256

        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_features, 512, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 3, 3, 1),
            nn.BatchNorm1d(256),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_prob),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Dropout(self.drop_prob),
            nn.ReLU()
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        
        x = x.view(self.batchsize, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc1(x)

        x = self.softmax(x)
        return x

    def predict(self, x_data):
        x_data = self.forward(x_data)
        result = self.softmax(x_data)
        return result 
    

#####################################
# Declarations and Hyper-parameters #
#####################################
lr = 0.001
epoch_number = 50
batchsize = 256
momentum = 0.9
weight_decay = 0.000001
nb_batch = int(len(X_train)/batchsize)

scaler = StandardScaler()

C = Classifier()

C.train()
# 82.2
if use_cuda:
    C.cuda(0)

C_optimizer = optim.SGD(C.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# C_optimizer = optim.Adam(C.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

#########
# Train #
#########

X_train = scaler.fit_transform(X_train)

for epoch in range(epoch_number):
    for index in range(nb_batch):
        X = torch.FloatTensor(X_train[index*batchsize:(index+1)*batchsize])
        Y_oh = torch.Tensor(Y_train_oh[index*batchsize:(index+1)*batchsize])
        Y_oh = Y_oh.float()
        if use_cuda:
            X = X.cuda(0)
            Y_oh = Y_oh.cuda(0)
        output = C(X)
        if use_cuda:
            output = output.cuda(0)

        loss = criterion(output, Y_oh)

        # back propagation
        C_optimizer.zero_grad()
        loss.backward()
        C_optimizer.step()

    print("epoch", epoch)

PATH = "/home/02mjpark/ConvGAN/SAVE/bsmdl.pt"
torch.save(C.state_dict(), PATH)
joblib.dump(scaler, '/home/02mjpark/ConvGAN/SAVE/scaler1.pkl')
