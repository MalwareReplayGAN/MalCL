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

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_train, Y_train = get_ember_train_data(data_dir) # get the ember train data
Y_train_oh = oh(Y_train) # one hot encoding of Y_train
X_train, Y_train_oh = shuffle_data(X_train, Y_train_oh) 

####################
# Classifier Model #
####################

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.output_dim = 100
        self.drop_prob = 0.5

        
        self.fc1 = nn.Linear(self.input_features, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc1_drop = nn.Dropout(self.drop_prob)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc2_drop = nn.Dropout(self.drop_prob)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc3_drop = nn.Dropout(self.drop_prob)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(256, self.output_dim)
        self.fc4_bn = nn.BatchNorm1d(self.output_dim)
        self.fc4_drop = nn.Dropout(self.drop_prob)
        self.act4 = nn.ReLU()

        # self.fc4 = nn.Linear(256, 128)
        # self.fc4_bn = nn.BatchNorm1d(128)
        # self.fc4_drop = nn.Dropout(0.5)
        # self.act4 = nn.ReLU()

        # self.fc5 = nn.Linear(128, self.output_dim)
        # self.fc5_bn = nn.BatchNorm1d(self.output_dim)
        # self.fc5_drop = nn.Dropout(0.5)
        # self.act5 = nn.ReLU()

        self.softmax = nn.Softmax()

    def forward(self, x):
        
        x = x.view(-1, self.input_features)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_drop(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.fc2_drop(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = self.fc3_drop(x)
        x = self.act3(x)

        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = self.fc4_drop(x)
        x = self.act4(x)
        
        # x = self.fc5(x)
        # x = self.fc5_bn(x)
        # x = self.fc5_drop(x)
        # x = self.act5(x)

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
epoch_number = 100
batchsize = 64
# momentum = 0.9
# weight_decay = 0.000001
nb_batch = int(len(X_train)/batchsize)

scaler = StandardScaler()

C = Classifier()

C.train()
# 82.2
if use_cuda:
    C.cuda(0)

# C_optimizer = optim.SGD(C.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
C_optimizer = optim.Adam(C.parameters(), lr=lr)

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
