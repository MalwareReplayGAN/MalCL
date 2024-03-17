import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_ import get_ember_train_data, extract_100data
# from data_ import extract_100data
# from model import Classifier
import torch.optim as optim


# switch to False to use CPU

use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu");
# torch.manual_seed(0);


# Call the Ember Data

data_dir = '/home/02mjpark/continual-learning-malware/ember_data/EMBER_CL/EMBER_Class'
X_train, Y_train, Y_train_onehot = get_ember_train_data(data_dir)
X_train_100, Y_train_100, Y_train_100_onehot = extract_100data(X_train, Y_train)
# print(Y_train_100_onehot[0:3])
# print(Y_train_100_onehot.shape)

####################
# Classifier Model #
####################

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.input_channel = 1
        self.channel_b = 128
        self.channel_c = 256
        self.channel_d = 512
        self.output_dim = 100
        self.drop_prob = 0.3

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_channel, self.channel_d, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.channel_d),
            nn.Dropout(self.drop_prob),
            nn.Conv1d(self.channel_d, self.channel_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.channel_c),
            nn.Dropout(self.drop_prob),
            nn.Conv1d(self.channel_c, self.channel_b, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.channel_b),
            nn.Dropout(self.drop_prob),
            nn.Flatten())

        self.fc = nn.Linear(self.channel_b * self.input_features, self.output_dim)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, self.input_channel, self.input_features)
        # print("1-", x.shape)
        x = self.conv(x)
        # print("2-", x.shape)
        x = x.view(-1, self.channel_b * self.input_features)
        # print("3-", x.shape)
        x = self.fc(x)
        x = self.softmax(x)
        # print("5-", x.shape)
        #x = x.view(-1, self.output_dim)
        # print("5-", x.shape)
        # print("C")
        return x

    def predict(self, x_data):
        x_data = self.forward(x_data)
        result = self.softmax(x_data)
        return result #가장 큰 인덱스 리턴
    

#####################################
# Declarations and Hyper-parameters #
#####################################
lr = 0.001
epoch_number = 10
batchsize = 16
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
        X_100 = torch.FloatTensor(X_train_100[index*batchsize:(index+1)*batchsize])
        Y_100_oh = torch.Tensor(Y_train_100_onehot[index*batchsize:(index+1)*batchsize])
        Y_100_oh = Y_100_oh.float()
        if use_cuda:
            X_100 = X_100.cuda(0)
            Y_100_oh = Y_100_oh.cuda(0)
        output = C(X_100)
        if use_cuda:
            output = output.cuda(0)
        loss = criterion(output, Y_100_oh)
        loss.backward()
        C_optimizer.step()
    print("epoch", epoch)

PATH = "/home/02mjpark/ConvGAN/SAVE/bsmdl.pt"
torch.save(C.state_dict(), PATH)
# device = torch.device('cpu')


# torch.cuda.empty_cache()
# del X_train, X_train_100, Y_train, Y_train_onehot, Y_train_100_onehot

# def test(model, x_test, y_test):

#     x_test = torch.FloatTensor(x_test)
#     y_test = torch.LongTensor(y_test)
#     # print(x_test.shape)
#     # print(y_test.shape)
#     # use_cuda = False
#     if use_cuda:
#         x_test = x_test.cuda(0)
#         # y_test = y_test.cuda(0)
#         model = model.cuda(0)
#     model.eval()
#     prediction = model(x_test)
#     predicted_classes = prediction.max(1)[1]
#     correct_count = (predicted_classes == y_test).sum().item()
#     cost = F.cross_entropy(prediction, Y_test_onehot)

#     print('Accuracy: {}% Cost: {:.6f}'.format(
#         correct_count / len(y_test) * 100, cost.item()
#     ))    

# with torch.no_grad():
#     test(C, X_test, Y_test)
