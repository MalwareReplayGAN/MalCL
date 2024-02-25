import numpy as np
import os
import os.path as opth
import argparse
import torch
import torch.nn as nn

class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self):
        super(Generator, self).__init__()

        self.latent_dim = 1024
        self.latent_ = 28
        self.input_dim = 62
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=self.latent_dim, kernel_size=3),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(self.latent_dim, 256, 3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, self.output_dim, 3),
            nn.Sigmoid(),
        )

        self.Sigmoid = nn.Sigmoid()
        self.apply(self.weights_init)

    def reinit(self):
      self.apply(self.weights_init)

    def forward(self, input):
        input = input.view(self.input_dim)
        x = self.fc(input)
        x = x.view(self.latent_dim)
        x = self.deconv(x)
        return x

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.input_features = 2381
        self.latent_dim = 1024

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(0.2),
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
        x = self.conv(input)
        x = x.view(128)
        return self.fc(x)
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_height = 28
        self.input_dim = 1
        self.output_dim = 10
        self.batchsize = 64

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim, self.batchsize, kernel_size=3),
            nn.Flatten())
        self.fc = nn.Linear(self.batchsize, self.output_dim)

    def forward(self, x):
        x = x.view(self.input_features)
        x = self.conv(x)
        x = self.fc(x)
        return x

    def predict(self, x_data):
        z=self.forward(x_data)
        return torch.argmax(z,axis=1) #가장 큰 인덱스 리턴
