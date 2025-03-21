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
        x = self.conv(input)
        x = self.fc(x)
        x = x.view(-1, self.channel_g, 1)
        x = self.deconv(x) 
        x = x.view(-1, self.output_features)
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
        x = self.conv(x)
        feature = x.view(-1, self.channel_c * self.input_features)
        x = self.fc(feature)
        return x.view(-1, 1), feature
    
class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.output_dim = 50
        self.drop_prob = 0.5

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
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_prob),
            nn.ReLU()
        )
        
        self.fc1_f = nn.Flatten()
        self.fc1 = nn.Linear(128, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        self.fc1_drop1 = nn.Dropout(self.drop_prob)
        self.fc1_act1 = nn.ReLU()
        

        self.softmax = nn.Softmax()

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

        x = self.softmax(x)   

        # If testing, reshape the output tensor back to the original shape
        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)

        return x

    def expand_output_layer(self, init_classes, nb_inc, task):

        old_fc1 = self.fc1
        old_fc1_bn1 = self.fc1_bn1
        self.output_dim = init_classes + nb_inc * task

        self.fc1 = nn.Linear(old_fc1.in_features, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)

        with torch.no_grad():
            self.fc1.weight[:old_fc1.out_features].copy_(old_fc1.weight.data)
            self.fc1.bias[:old_fc1.out_features].copy_(old_fc1.bias.data)
            self.fc1_bn1.weight[:old_fc1_bn1.num_features].copy_(old_fc1_bn1.weight.data)
            self.fc1_bn1.bias[:old_fc1_bn1.num_features].copy_(old_fc1_bn1.bias.data)
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

        logits = self.fc1_f(x)
        
        return logits.detach()
