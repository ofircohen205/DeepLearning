# Name: Ofir Cohen
# ID: 312255847
# Date: 25/6/2020

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

class OneFC(nn.Module):

    def __init__(self):
        super(OneFC, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=32 * 32 * 3, out_features=10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc1(x)
        return x
''' End class '''
