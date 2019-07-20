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
        
        self.fc21 = nn.Linear(975, 1024)
        self.bn21 = nn.BatchNorm1d(1024)
        self.fc31 = nn.Linear(1024, 512)
        self.bn31 = nn.BatchNorm1d(512)
        self.fc41 = nn.Linear(512, 256)
        self.bn41 = nn.BatchNorm1d(256)
        self.fc51 = nn.Linear(256, 128)
        self.bn51 = nn.BatchNorm1d(128)
        self.fc61 = nn.Linear(128, 64)
        self.bn61 = nn.BatchNorm1d(64)
        self.fc71 = nn.Linear(64, 32)
        self.bn71 = nn.BatchNorm1d(32)
        self.fc81 = nn.Linear(32, 16)
        self.bn81 = nn.BatchNorm1d(16)
			
        self.fc9 = nn.Linear(32, 16)
        self.bn9 = nn.BatchNorm1d(16)
        self.fc10 = nn.Linear(16, 1)
        
        self.conv1s  = nn.Conv1d(1,32,5)
        self.avg1s   = nn.AvgPool1d(2)		  
        self.conv2s  = nn.Conv1d(32,3,5)
        self.avg2s   = nn.AvgPool1d(2)	
        
        self.conv3s  = nn.Conv1d(1,32,7)
        self.avg3s   = nn.AvgPool1d(2)		  
        self.conv4s  = nn.Conv1d(32,3,7)
        self.avg4s   = nn.AvgPool1d(2)		  
		  
    def forward(self, s):
        fs = self.fc1(s)
        s = s.unsqueeze(1)
        xs1 = self.avg2s(self.conv2s(self.avg1s(self.conv1s(s))))
        xs1 = xs1.view(xs1.shape[0], -1)
        xs2 = self.avg4s(self.conv4s(self.avg3s(self.conv3s(s))))
        xs2 = xs2.view(xs2.shape[0], -1)
        xs = torch.cat((xs1,xs2),dim=1)
        s = F.relu(self.bn2(self.fc2(fs)))
        s = F.relu(self.bn3(self.fc3(s)))
        s = F.relu(self.bn4(self.fc4(s)))
        s = F.relu(self.bn5(self.fc5(s)))
        s = F.relu(self.bn6(self.fc6(s)))
        s = F.relu(self.bn7(self.fc7(s)))
        s = F.relu(self.bn8(self.fc8(s)))
        
        xs = F.relu(self.bn21(self.fc21(xs)))
        xs = F.relu(self.bn31(self.fc31(xs)))
        xs = F.relu(self.bn41(self.fc41(xs)))
        xs = F.relu(self.bn51(self.fc51(xs)))
        xs = F.relu(self.bn61(self.fc61(xs)))
        xs = F.relu(self.bn71(self.fc71(xs)))
        xs = F.relu(self.bn81(self.fc81(xs)))
        
        s = torch.cat((s,xs),dim=1)
        s = F.relu(self.bn9(self.fc9(s)))
        s = self.fc10(s)
        return s