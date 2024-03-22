import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_ import get_ember_train_data, extract_100data, shuffle_data, oh
import torch.optim as optim


# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu");
# torch.manual_seed(0);


# Call the Ember Data

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_train, Y_train = get_ember_train_data(data_dir) # get the ember train data
X_train_100, Y_train_100 = extract_100data(X_train, Y_train) # extrach 100 data from each label
Y_train_oh = oh(Y_train) # one hot encoding of Y_train
Y_train_100_oh = oh(Y_train_100) # one hot encoding of Y_train_100
X_train_100, Y_train_100_oh = shuffle_data(X_train_100, Y_train_100_oh) 

####################
# Classifier Model #
####################

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        # self.input_channel = 1
        self.output_dim = 100
        self.drop_prob = 0.5

        # self.conv = nn.Sequential(
        #     nn.Conv1d(self.input_channel, self.channel_d, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.channel_d),
        #     nn.Dropout(self.drop_prob),
        #     nn.Conv1d(self.channel_d, self.channel_c, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.channel_c),
        #     nn.Dropout(self.drop_prob),
        #     nn.Conv1d(self.channel_c, self.channel_b, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.channel_b),
        #     nn.Dropout(self.drop_prob),
        #     nn.Flatten())

        # self.fc1 =  nn.Sequential(
        #     nn.Linear(self.input_features, 3000),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(3000),
        #     nn.Dropout(self.drop_prob),
        #     nn.Linear(3000, 4000),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(4000),
        #     nn.Dropout(self.drop_prob),
        #     nn.Linear(4000, 1000),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(1000),
        #     nn.Linear(1000, self.input_features),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.input_features)
        # )

        # self.fc2 = nn.Linear(self.input_features, self.output_dim)

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

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, self.input_features)
        # x = x.view(-1, self.input_channel, self.input_features)
        # x = self.conv(x)
        # x = x.view(-1, self.channel_b * self.input_features)
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
epoch_number = 500
batchsize = 16
# nb_batch = int(len(X_train)/batchsize)
nb_batch = int(len(X_train_100)/batchsize)

C = Classifier()

C.train()

if use_cuda:
    C.cuda(0)

C_optimizer = optim.Adam(C.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

#########
# Train #
#########

for epoch in range(epoch_number):
    for index in range(nb_batch):
        C_optimizer.zero_grad()
        # X_100 = torch.FloatTensor(X_train[index*batchsize:(index+1)*batchsize])
        X_100 = torch.FloatTensor(X_train_100[index*batchsize:(index+1)*batchsize])
        # Y_100_oh = torch.Tensor(Y_train_onehot[index*batchsize:(index+1)*batchsize])
        # print(Y_train_100_oh)
        
        Y_100_oh = torch.Tensor(Y_train_100_oh[index*batchsize:(index+1)*batchsize])
        Y_100_oh = Y_100_oh.float()
        if use_cuda:
            X_100 = X_100.cuda(0)
            Y_100_oh = Y_100_oh.cuda(0)
        output = C(X_100)
        if use_cuda:
            output = output.cuda(0)
        # print(output.shape, Y_100_oh.shape)
        loss = criterion(output, Y_100_oh)
        loss.backward()
        C_optimizer.step()
    print("epoch", epoch)

PATH = "/home/02mjpark/ConvGAN/SAVE/bsmdl_100.pt"
torch.save(C.state_dict(), PATH)
