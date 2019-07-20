import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(666, 20)
        self.fc2 = nn.Linear(20, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.fc8 = nn.Linear(32, 16)
        self.bn8 = nn.BatchNorm1d(16)
        self.fc9 = nn.Linear(16, 1)
		  
    def forward(self, s):
        s = self.fc1(s)
        s = F.relu(self.bn2(self.fc2(s)))
        s = F.relu(self.bn3(self.fc3(s)))
        s = F.relu(self.bn4(self.fc4(s)))
        s = F.relu(self.bn5(self.fc5(s)))
        s = F.relu(self.bn6(self.fc6(s)))
        s = F.relu(self.bn7(self.fc7(s)))
        s = F.relu(self.bn8(self.fc8(s)))
        s = self.fc9(s)
        
        return s