import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()
        
        self.fc10 = nn.Linear(666, 52)
        self.fc11 = nn.Linear(666, 52)
        self.fc12 = nn.Linear(666, 52)
        self.fc13 = nn.Linear(666, 52)
        self.fc14 = nn.Linear(666, 52)
        self.fc15 = nn.Linear(666, 52)
        self.fc16 = nn.Linear(666, 52)
        self.fc17 = nn.Linear(666, 52)
        self.fc18 = nn.Linear(666, 52)
        self.fc19 = nn.Linear(666, 52)
        
                
        
        self.fc20 = nn.Linear(30, 128)
        self.t1bn20 = nn.BatchNorm1d(128)
        self.t1fc2 = nn.Linear(128, 256)
        self.t1bn2 = nn.BatchNorm1d(256)
        self.t1fc3 = nn.Linear(256, 512)
        self.t1bn3 = nn.BatchNorm1d(512)
        self.t1fc4 = nn.Linear(512, 1024)
        self.t1bn4 = nn.BatchNorm1d(1024)
        self.t1fc5 = nn.Linear(1024, 768)
        self.t1bn5 = nn.BatchNorm1d(768)
        self.t1fc6 = nn.Linear(768, 640)
        self.t1bn6 = nn.BatchNorm1d(640)
        self.t1fc7 = nn.Linear(640, 512)
        self.t1bn7 = nn.BatchNorm1d(512)
        self.t1fc8 = nn.Linear(512, 384)
        self.t1bn8 = nn.BatchNorm1d(384)
        self.t1fc9 = nn.Linear(384, 256)
        self.t1bn9 = nn.BatchNorm1d(256)
        self.t1fc10 = nn.Linear(256, 192)
        self.t1bn10 = nn.BatchNorm1d(192)
        self.t1fc11 = nn.Linear(192, 128)
        self.t1bn11 = nn.BatchNorm1d(128)
        self.t1fc12 = nn.Linear(128, 40)
        self.t1bn12 = nn.BatchNorm1d(40)
        self.t1fc13 = nn.Linear(40, 16)
        self.t1bn13  = nn.BatchNorm1d(16)
        self.t1fc14 = nn.Linear(16, 8)
        self.t1bn14 = nn.BatchNorm1d(8)
        self.t1fc15 = nn.Linear(8, 1)
        

        
        self.conv1  = nn.Conv1d(10,16,5)
        self.avg1   = nn.AvgPool1d(2)		  

        self.conv2  = nn.Conv1d(16,3,5)
        self.avg2   = nn.AvgPool1d(2)		  

		  
    def forward(self, s):
        fs0 = self.fc10(s)
        fs1 = self.fc11(s)
        fs2 = self.fc12(s)
        fs3 = self.fc13(s)
        fs4 = self.fc14(s)
        fs5 = self.fc15(s)
        fs6 = self.fc16(s)
        fs7 = self.fc17(s)
        fs8 = self.fc18(s)
        fs9 = self.fc19(s)
        
        fs0 = fs0.unsqueeze(1)
        fs1 = fs1.unsqueeze(1)
        fs2 = fs2.unsqueeze(1)
        fs3 = fs3.unsqueeze(1)
        fs4 = fs4.unsqueeze(1)
        fs5 = fs5.unsqueeze(1)
        fs6 = fs6.unsqueeze(1)
        fs7 = fs7.unsqueeze(1)
        fs8 = fs8.unsqueeze(1)
        fs9 = fs9.unsqueeze(1)

			
        s = torch.cat((fs0,fs1,fs2,fs3,fs4,fs5,fs6,fs7,fs8,fs9),dim=1)

        s = self.avg2(self.conv2(self.avg1(self.conv1(s))))

        s = s.view(s.shape[0], -1)

        t1s128 = self.t1fc20(s)
        t1s256= self.t1fc2(F.relu(self.t1bn20(t1s128)))
        t1s512 = self.t1fc3(F.relu(self.t1bn2(t1s256)))
        t1s= F.relu(self.t1bn4(self.t1fc4(F.relu(self.t1bn3(t1s512)))))
        t1s= F.relu(self.t1bn5(self.t1fc5(t1s)))
        t1s= F.relu(self.t1bn6(self.t1fc6(t1s)))
        t1s= self.t1fc7(t1s)
        t1s= F.relu(self.t1bn8(self.t1fc8(F.relu(self.t1bn7(t1s + t1s512)))))
        t1s= self.t1fc9(t1s)
        t1s= F.relu(self.t1bn10(self.t1fc10(F.relu(self.t1bn9(t1s+t1s256)))))
        t1s= self.t1fc11(t1s)
        t1s= self.t1fc12(F.relu(self.t1bn11(t1s+t1s128)))
        t1s= F.relu(self.t1bn13(self.t1fc13(F.relu(self.t1bn12(t1s+s)))))
        t1s= F.relu(self.t1bn14(self.t1fc14(t1s)))
        t1s= self.t1fc15(t1s)
        
        return t1s