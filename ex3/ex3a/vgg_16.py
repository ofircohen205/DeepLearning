# Name: Ofir Cohen
# ID: 312255847
# Date: 15/5/2020

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import math, os
from load_cifar10 import load_cifar_10_data, load_cifar_10_data_tf

class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x
''' End class '''


def vgg16_bn():
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(make_layers(config, True))
''' End function '''


def make_layers(config, batch_norm=False):
    layers = []
    in_channels = 3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)
''' End function '''