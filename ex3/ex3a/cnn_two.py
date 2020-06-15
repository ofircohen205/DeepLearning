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

class CnnTwo(nn.Module):

    def __init__(self, features):
        super(CnnTwo, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(64 * 64 * 2, 200),
            nn.ReLU(True),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
''' End class '''


def cnn_two_bn():
    config = [64, 'M', 128, 'M']
    return CnnTwo(make_layers(config, True))
''' End function '''


def make_layers(config, batch_norm=False):
    layers = []
    in_channels = 3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)
''' End function '''