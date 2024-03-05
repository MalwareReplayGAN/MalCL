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
        self.output_channel_1 = 256
        self.output_channel_2 = 512
        self.output_features = 2381

        self.fc = nn.Sequential(
            nn.Conv1d(self.input_dim, self.output_channel_1, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.output_channel_1),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(self.output_channel_1, self.output_channel_2, 3, padding=1),
            nn.BatchNorm1d(self.output_channel_2),
            nn.ReLU(),
            nn.ConvTranspose1d(self.output_channel_2, self.output_features, 3, padding=1),
            nn.Sigmoid(),
        )

        self.Sigmoid = nn.Sigmoid()
        self.apply(self.weights_init)

    def reinit(self):
        self.apply(self.weights_init)

    def forward(self, input):
        input = input.view(-1, self.input_dim, 1)
        # print("1-", input.shape) # [batchsize, self.input_dim, 1] 16, 62, 1
        x = self.fc(input)
        # print("2-", x.shape) # [batchsize, self.output_channel, 1]16, 256, 1
        # x = x.view(-1, self.output_channel_1, 1)
        # print("3-", x.shape) # [batchsize, self.output_channel]
        x = self.deconv(x) # Given transposed=1, weight of size [self.output_channel]
        # print("4-", x.shape) # 16, 1, 1
        return x.view(-1, 1)

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
        self.output_channel_1 = 256
        self.output_channel_2 = 512
        self.input_features = 2381
        self.latent_dim = 1024

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_channel, self.output_channel_1, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.output_channel_1, self.output_channel_2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.output_channel_2),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(self.output_channel_2 * self.input_features, self.latent_dim),
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
        x = x.view(-1, self.output_channel_2 * self.input_features)
        # print("3-", x.shape) # [16, 512*2381]
        x = self.fc(x)
        # print("4-", x.shape) # [16, 1]
        return x.view(-1, 1)
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.input_channel = 1
        self.output_channel = 128
        self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_channel, self.output_channel, kernel_size=3, padding=1),
            nn.Flatten())

        self.fc = nn.Linear(self.output_channel * self.input_features, self.output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_channel, self.input_features)
        # print("1-", x.shape)
        x = self.conv(x)
        # print("2-", x.shape)
        x = x.view(-1, self.output_channel * self.input_features)
        # print("3-", x.shape)
        x = self.fc(x)
        # print("4-", x.shape)
        x = x.view(-1, self.output_dim)
        # print("5-", x.shape)
        # print("C")
        return x

    def predict(self, x_data):
        z=self.forward(x_data)
        return torch.argmax(z,axis=1) #가장 큰 인덱스 리턴



# data_size = (16, 1, 2381) # 배치 크기, 채널, 시퀀스 길이
# conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3) out_channels는 출력 데이터의 채널 개수. 컨볼루션 필터의 개수를 의미하며, 출력데이터가 몇 개의 특징 맵으로 변환되는지