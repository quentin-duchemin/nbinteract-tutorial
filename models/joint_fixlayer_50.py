import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()

        
        self.t1fc2 = nn.Linear(50, 2048)
        self.t1bn2 = nn.BatchNorm1d(2048)
        self.t1fc3 = nn.Linear(2048, 1536)
        self.t1bn3 = nn.BatchNorm1d(1536)
        self.t1fc4 = nn.Linear(1536, 1024)
        self.t1bn4 = nn.BatchNorm1d(1024)
        self.t1fc5 = nn.Linear(1024, 512)
        self.t1bn5 = nn.BatchNorm1d(512)
        self.t1fc6 = nn.Linear(512, 256)
        self.t1bn6= nn.BatchNorm1d(256)
        self.t1fc7 = nn.Linear(256, 128)
        self.t1bn7 = nn.BatchNorm1d(128)
        self.t1fc8 = nn.Linear(128, 64)
        self.t1bn8 = nn.BatchNorm1d(64)
        self.t1fc9 = nn.Linear(64, 32)
        self.t1bn9 = nn.BatchNorm1d(32)
        self.t1fc10 = nn.Linear(32, 16)
        self.t1bn10= nn.BatchNorm1d(16)
        self.t1fc11= nn.Linear(16, 1)


        self.t2fc2 = nn.Linear(50, 2048)
        self.t2bn2 = nn.BatchNorm1d(2048)
        self.t2fc3 = nn.Linear(2048, 1536)
        self.t2bn3 = nn.BatchNorm1d(1536)
        self.t2fc4 = nn.Linear(1536, 1024)
        self.t2bn4 = nn.BatchNorm1d(1024)
        self.t2fc5 = nn.Linear(1024, 512)
        self.t2bn5 = nn.BatchNorm1d(512)
        self.t2fc6 = nn.Linear(512, 256)
        self.t2bn6= nn.BatchNorm1d(256)
        self.t2fc7 = nn.Linear(256, 128)
        self.t2bn7 = nn.BatchNorm1d(128)
        self.t2fc8 = nn.Linear(128, 64)
        self.t2bn8 = nn.BatchNorm1d(64)
        self.t2fc9 = nn.Linear(64, 32)
        self.t2bn9 = nn.BatchNorm1d(32)
        self.t2fc10 = nn.Linear(32, 16)
        self.t2bn10= nn.BatchNorm1d(16)
        self.t2fc11= nn.Linear(16, 1)


        self.m0sfc2 = nn.Linear(50, 2048)
        self.m0sbn2 = nn.BatchNorm1d(2048)
        self.m0sfc3 = nn.Linear(2048, 1536)
        self.m0sbn3 = nn.BatchNorm1d(1536)
        self.m0sfc4 = nn.Linear(1536, 1024)
        self.m0sbn4 = nn.BatchNorm1d(1024)
        self.m0sfc5 = nn.Linear(1024, 512)
        self.m0sbn5 = nn.BatchNorm1d(512)
        self.m0sfc6 = nn.Linear(512, 256)
        self.m0sbn6= nn.BatchNorm1d(256)
        self.m0sfc7 = nn.Linear(256, 128)
        self.m0sbn7 = nn.BatchNorm1d(128)
        self.m0sfc8 = nn.Linear(128, 64)
        self.m0sbn8 = nn.BatchNorm1d(64)
        self.m0sfc9 = nn.Linear(64, 32)
        self.m0sbn9 = nn.BatchNorm1d(32)
        self.m0sfc10 = nn.Linear(32, 16)
        self.m0sbn10= nn.BatchNorm1d(16)
        self.m0sfc11= nn.Linear(16, 1)
        
    def forward(self, fs):


        # T1
        t1s = F.relu(self.t1bn2(self.t1fc2(fs)))
        t1s = F.relu(self.t1bn3(self.t1fc3(t1s)))
        t1s = F.relu(self.t1bn4(self.t1fc4(t1s)))
        t1s = F.relu(self.t1bn5(self.t1fc5(t1s)))
        t1s = F.relu(self.t1bn6(self.t1fc6(t1s)))
        t1s = F.relu(self.t1bn7(self.t1fc7(t1s)))
        t1s = F.relu(self.t1bn8(self.t1fc8(t1s)))
        t1s = F.relu(self.t1bn9(self.t1fc9(t1s)))
        t1s = F.relu(self.t1bn10(self.t1fc10(t1s)))
        t1s = self.t1fc11(t1s)

        t2s = F.relu(self.t2bn2(self.t2fc2(fs)))
        t2s = F.relu(self.t2bn3(self.t2fc3(t2s)))
        t2s = F.relu(self.t2bn4(self.t2fc4(t2s)))
        t2s = F.relu(self.t2bn5(self.t2fc5(t2s)))
        t2s = F.relu(self.t2bn6(self.t2fc6(t2s)))
        t2s = F.relu(self.t2bn7(self.t2fc7(t2s)))
        t2s = F.relu(self.t2bn8(self.t2fc8(t2s)))
        t2s = F.relu(self.t2bn9(self.t2fc9(t2s)))
        t2s = F.relu(self.t2bn10(self.t2fc10(t2s)))
        t2s = self.t2fc11(t2s)


        m0ss = F.relu(self.m0sbn2(self.m0sfc2(fs)))
        m0ss = F.relu(self.m0sbn3(self.m0sfc3(m0ss)))
        m0ss = F.relu(self.m0sbn4(self.m0sfc4(m0ss)))
        m0ss = F.relu(self.m0sbn5(self.m0sfc5(m0ss)))
        m0ss = F.relu(self.m0sbn6(self.m0sfc6(m0ss)))
        m0ss = F.relu(self.m0sbn7(self.m0sfc7(m0ss)))
        m0ss = F.relu(self.m0sbn8(self.m0sfc8(m0ss)))
        m0ss = F.relu(self.m0sbn9(self.m0sfc9(m0ss)))
        m0ss = F.relu(self.m0sbn10(self.m0sfc10(m0ss)))
        m0ss = self.m0sfc11(m0ss)

        # final prediction
        result = torch.cat((m0ss,t1s,t2s),dim=1)
        
        return result
        
        