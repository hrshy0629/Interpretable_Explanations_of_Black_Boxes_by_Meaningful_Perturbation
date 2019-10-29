# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

class lenet(nn.Module):
    def __init__(self, input, kernel):#, kernel, stride, padding, input, classes
        super(lenet, self).__init__()
        self.kernel = kernel
        self.input = input
        #Conv2d(inout_filter, output_filter, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(self.input, 6, self.kernel, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, self.kernel, stride=1, padding=0)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        #x = x[:, np.newaxis, :, :]
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
