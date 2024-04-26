import numpy as np
import os
import os.path as opth
import argparse
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.input_dim = 62
        self.channel_a = 64
        self.channel_b = 128
        self.channel_c = 256
        self.channel_d = 512
        self.channel_e = 1024
        self.channel_f = 2048
        self.channel_g = 4096
        self.output_features = 2381


        
        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim, self.channel_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channel_c),
            nn.ReLU(),
            nn.Conv1d(self.channel_c, self.channel_e, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channel_e),
            nn.ReLU(),
            nn.Conv1d(self.channel_e, self.channel_g, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channel_g),
            nn.ReLU(),
            nn.Conv1d(self.channel_g, self.channel_e, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.channel_e),
            nn.ReLU(),
            
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channel_e, self.channel_f), 
            nn.BatchNorm1d(self.channel_f),
            nn.ReLU(),
            nn.Linear(self.channel_f, self.channel_g), 
            nn.BatchNorm1d(self.channel_g),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(self.channel_g, self.channel_e, 3, padding=1),
            nn.BatchNorm1d(self.channel_e),
            nn.ReLU(),
            nn.ConvTranspose1d(self.channel_e, self.channel_d, 3, padding=1),
            nn.BatchNorm1d(self.channel_d),
            nn.ReLU(),
            nn.ConvTranspose1d(self.channel_d, self.output_features, 3, padding=1),
            nn.Sigmoid()
        )

        self.Sigmoid = nn.Sigmoid()
        self.apply(self.weights_init)

    def reinit(self):
        self.apply(self.weights_init)

    def forward(self, input):
        input = input.view(-1, self.input_dim, 1)
        # print("1-", input.shape) # [batchsize, self.input_dim, 1] 16, 62, 1
        x = self.conv(input)
        # print("2-", x.shape) # [batchsize, self.output_channel, 1]16, 512, 1
        x = self.fc(x)
        x = x.view(-1, self.channel_g, 1)
        # print("3-", x.shape) # [batchsize, self.output_channel]16, 512, 1
        x = self.deconv(x) # Given transposed=1, weight of size [self.output_channel]
        # print("4-", x.shape) # 16, 2381, 1
        x = x.view(-1, self.output_features)
        # print("5-", x.shape) # 16, 2381
        return x

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.input_channel = 1
        self.output_dim = 1
        self.channel_c = 256
        self.channel_d = 512
        self.input_features = 2381
        self.latent_dim = 1024

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_channel, self.channel_d, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.channel_d, self.channel_c, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.channel_c),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(self.channel_c * self.input_features, self.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim),
            nn.Linear(self.latent_dim, self.output_dim),
            nn.Sigmoid(),
        )

        self.apply(self.weights_init)

    def reinit(self):
      self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, input):
        x = input.view(-1, self.input_channel, self.input_features)
        # print("1-", x.shape) # [16, 1, 2381]
        x = self.conv(x)
        # print("2-", x.shape) # [16, 512, 2381]
        x = x.view(-1, self.channel_c * self.input_features)
        # print("3-", x.shape) # [16, 512*2381]
        x = self.fc(x)
        # print("4-", x.shape) # [16, 1]
        return x.view(-1, 1)
    
class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.input_channel = 1
        self.output_dim = 20
        self.output_dim = 20
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
        
        self.fc4 = nn.Linear(256, 128)
        self.fc4_bn = nn.BatchNorm1d(128)
        self.fc4_drop = nn.Dropout(self.drop_prob)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(128, self.output_dim)
        self.fc5_bn = nn.BatchNorm1d(self.output_dim)
        self.fc5_drop = nn.Dropout(0.5)
        self.act5 = nn.ReLU()

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
        
        x = self.fc5(x)
        x = self.fc5_bn(x)
        x = self.fc5_drop(x)
        x = self.act5(x)

        x = self.softmax(x)
    
        return x

    def expand_output_layer(self, init_classes, nb_inc, task):
        """
        Expand the output layer to accommodate new_classes.
        This method retains the weights of the existing layer and expands it to fit the new class count.
        """
        old_fc5 = self.fc5
        self.output_dim = init_classes + nb_inc * task

        # Create a new classifier layer with the updated output dimension.
        self.fc5 = nn.Linear(128, self.output_dim)
        self.fc5_bn = nn.BatchNorm1d(self.output_dim)

        # Trnasfer the old weights and biases
        with torch.no_grad():
            self.fc5.weight[:old_fc5.out_features].copy_(old_fc5.weight.data)
            self.fc5.bias[:old_fc5.out_features].copy_(old_fc5.bias.data)

        return self
    
    # def update_output_dim(self, init_classes, nb_inc, task):
    #     self.output_dim = init_classes + nb_inc * task
    #     return self.output_dim

    def predict(self, x_data):
        result = self.forward(x_data)
        result = self.softmax(result)
        return result
        # return torch.argmax(z,axis=1) #가장 큰 인덱스 리턴



# data_size = (16, 1, 2381) # 배치 크기, 채널, 시퀀스 길이
# conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3) out_channels는 출력 데이터의 채널 개수. 컨볼루션 필터의 개수를 의미하며, 출력데이터가 몇 개의 특징 맵으로 변환되는지