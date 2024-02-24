import numpy as np
import os
import os.path as opth
import argparse
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.laten_T)
        )
    def conv1d_block():
        nn.Conv1d(in_channels=, out_channels=, kernel_size=, stride=, padding=0, )
    def reinit(self):
        self.apply(self.weights_init)

    def forward(self, input):
        input = input.
        x = self.fc(input)
        x = 
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    def conv1d_
    def reinit(self):
        self.apply(self.weights_init)
    def weight_init(self, m):

    
class Classifier:
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.in_channels = 
        self.out_channels = 
        

        self.conv = nn.Sequential(
            nn.Conv1d(self.in_channels=, self.out_channels=, kernel_size=),
            nn.
        )
        self.fc = nn.Linear
    def forward(self, ):
        input = 
        x = self.
    def predict(self, x_data):
        z = self.forward(x_data)
        return torch.argmax(z, axis=1) # returning the biggest index
