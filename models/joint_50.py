import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(666, 50)
        
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
        
    def forward(self, signal):
        fs = self.fc1(signal)

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
        
        



#   self.t2fc2 = nn.Linear(20, 1024)
#   self.t2bn2 = nn.BatchNorm1d(1024)
#   self.t2fc3 = nn.Linear(1024, 512)
#   self.t2bn3 = nn.BatchNorm1d(512)
#   self.t2fc4 = nn.Linear(512, 256)
#   self.t2bn4 = nn.BatchNorm1d(256)
#   self.t2fc5 = nn.Linear(256, 128)
#   self.t2bn5 = nn.BatchNorm1d(128)
#   self.t2fc6 = nn.Linear(128, 64)
#   self.t2bn6 = nn.BatchNorm1d(64)
#   self.t2fc7 = nn.Linear(64, 32)
#   self.t2bn7 = nn.BatchNorm1d(32)
#   self.t2fc8 = nn.Linear(32, 16)
#   self.t2bn8 = nn.BatchNorm1d(16)
#   self.t2fc9 = nn.Linear(16, 1)


#   self.m0sfc10 = nn.Linear(666, 52)
#   self.m0sfc11 = nn.Linear(666, 52)
#   self.m0sfc12 = nn.Linear(666, 52)
#   self.m0sfc13 = nn.Linear(666, 52)
#   self.m0sfc14 = nn.Linear(666, 52)
#   self.m0sfc15 = nn.Linear(666, 52)
#   self.m0sfc16 = nn.Linear(666, 52)
#   self.m0sfc17 = nn.Linear(666, 52)
#   self.m0sfc18 = nn.Linear(666, 52)
#   self.m0sfc19 = nn.Linear(666, 52)
#   self.m0sfc3 = nn.Linear(30, 512)
#   self.m0sfc4 = nn.Linear(512, 256)
#   self.m0sfc5 = nn.Linear(256, 128)
#   self.m0sfc6 = nn.Linear(128, 64)
#   self.m0sfc7 = nn.Linear(64, 32)
#   self.m0sfc8 = nn.Linear(32, 16)
#   self.m0sfc9 = nn.Linear(16, 1)
#   self.m0sconv1  = nn.Conv1d(10,16,5)
#   self.m0savg1   = nn.AvgPool1d(2)         
#   self.m0sconv2  = nn.Conv1d(16,3,5)
#   self.m0savg2   = nn.AvgPool1d(2)   
		  
		  


#   # T2
#   t2s = F.relu(self.t2bn2(self.t2fc2(fs)))
#   t2s = F.relu(self.t2bn3(self.t2fc3(t2s)))
#   t2s = F.relu(self.t2bn4(self.t2fc4(t2s)))
#   t2s = F.relu(self.t2bn5(self.t2fc5(t2s)))
#   t2s = F.relu(self.t2bn6(self.t2fc6(t2s)))
#   t2s = F.relu(self.t2bn7(self.t2fc7(t2s)))
#   t2s = F.relu(self.t2bn8(self.t2fc8(t2s)))
#   t2s = self.t2fc9(t2s)

#   # m0s
#   fs0 = self.m0sfc10(signal)
#   fs1 = self.m0sfc11(signal)
#   fs2 = self.m0sfc12(signal)
#   fs3 = self.m0sfc13(signal)
#   fs4 = self.m0sfc14(signal)
#   fs5 = self.m0sfc15(signal)
#   fs6 = self.m0sfc16(signal)
#   fs7 = self.m0sfc17(signal)
#   fs8 = self.m0sfc18(signal)
#   fs9 = self.m0sfc19(signal)
  
#   fs0 = fs0.unsqueeze(1)
#   fs1 = fs1.unsqueeze(1)
#   fs2 = fs2.unsqueeze(1)
#   fs3 = fs3.unsqueeze(1)
#   fs4 = fs4.unsqueeze(1)
#   fs5 = fs5.unsqueeze(1)
#   fs6 = fs6.unsqueeze(1)
#   fs7 = fs7.unsqueeze(1)
#   fs8 = fs8.unsqueeze(1)
#   fs9 = fs9.unsqueeze(1)

   
#   m0ss = torch.cat((fs0,fs1,fs2,fs3,fs4,fs5,fs6,fs7,fs8,fs9),dim=1)

#   m0ss = self.m0savg2(self.m0sconv2(self.m0savg1(self.m0sconv1(m0ss))))

#   m0ss = m0ss.view(m0ss.shape[0], -1)

#   m0ss = F.relu((self.m0sfc3(m0ss)))
#   m0ss = F.relu((self.m0sfc4(m0ss)))
#   m0ss = F.relu((self.m0sfc5(m0ss)))
#   m0ss = F.relu((self.m0sfc6(m0ss)))
#   m0ss = F.relu((self.m0sfc7(m0ss)))
#   m0ss = F.relu((self.m0sfc8(m0ss)))
#   m0ss = self.m0sfc9(m0ss)
		   
		   
# T2
# net.t2fc2.weight.data = nett2.fc2.weight.data
# net.t2fc2.bias.data = nett2.fc1.bias.data
# net.t2fc3.weight.data = nett2.fc3.weight.data
# net.t2fc3.bias.data = nett2.fc3.bias.data
# net.t2fc4.weight.data = nett2.fc4.weight.data
# net.t2fc4.bias.data = nett2.fc4.bias.data
# net.t2fc5.weight.data = nett2.fc5.weight.data
# net.t2fc5.bias.data = nett2.fc5.bias.data
# net.t2fc6.weight.data = nett2.fc6.weight.data
# net.t2fc6.bias.data = nett2.fc6.bias.data
# net.t2fc7.weight.data = nett2.fc7.weight.data
# net.t2fc7.bias.data = nett2.fc7.bias.data
# net.t2fc8.weight.data = nett2.fc8.weight.data
# net.t2fc8.bias.data = nett2.fc8.bias.data
# net.t2fc9.weight.data = nett2.fc9.weight.data
# net.t2fc9.bias.data = nett2.fc9.bias.data

# net.t2bn2.weight.data = nett2.bn2.weight.data
# net.t2bn2.bias.data = nett2.bn1.bias.data
# net.t2bn3.weight.data = nett2.bn22.weight.data
# net.t2bn3.bias.data = nett2.bn22.bias.data
# net.t2bn4.weight.data = nett2.bn23.weight.data
# net.t2bn4.bias.data = nett2.bn23.bias.data
# net.t2bn5.weight.data = nett2.v3.weight.data
# net.t2bn5.bias.data = nett2.bn3.bias.data
# net.t2bn6.weight.data = nett2.bn4.weight.data
# net.t2bn6.bias.data = nett2.bn4.bias.data
# net.t2bn7.weight.data = nett2.bn5.weight.data
# net.t2bn7.bias.data = nett2.bn5.bias.data
# net.t2bn8.weight.data = nett2.bn6.weight.data
# net.t2bn8.bias.data = nett2.bn6.bias.data

# #m0s
# net.m0sfc10.weight.data = netm0s.fc10.weight.data
# net.m0sfc10.bias.data = netm0s.fc10.bias.data
# net.m0sfc11.weight.data = netm0s.fc11.weight.data
# net.m0sfc11.bias.data = netm0s.fc11.bias.data
# net.m0sfc12.weight.data = netm0s.fc12.weight.data
# net.m0sfc12.bias.data = netm0s.fc12.bias.data
# net.m0sfc13.weight.data = netm0s.fc13.weight.data
# net.m0sfc13.bias.data = netm0s.fc13.bias.data
# net.m0sfc14.weight.data = netm0s.fc14.weight.data
# net.m0sfc14.bias.data = netm0s.fc15.bias.data
# net.m0sfc15.weight.data = netm0s.fc15.weight.data
# net.m0sfc15.bias.data = netm0s.fc15.bias.data
# net.m0sfc16.weight.data = netm0s.fc16.weight.data
# net.m0sfc16.bias.data = netm0s.fc16.bias.data
# net.m0sfc17.weight.data = netm0s.fc17.weight.data
# net.m0sfc17.bias.data = netm0s.fc17.bias.data
# net.m0sfc18.weight.data = netm0s.fc18.weight.data
# net.m0sfc18.bias.data = netm0s.fc18.bias.data
# net.m0sfc19.weight.data = netm0s.fc19.weight.data
# net.m0sfc19.bias.data = netm0s.fc19.bias.data

# net.m0sfc3.weight.data = netm0s.fc3.weight.data
# net.m0sfc3.bias.data = netm0s.fc3.bias.data
# net.m0sfc4.weight.data = netm0s.fc4.weight.data
# net.m0sfc4.bias.data = netm0s.fc5.bias.data
# net.m0sfc5.weight.data = netm0s.fc5.weight.data
# net.m0sfc5.bias.data = netm0s.fc5.bias.data
# net.m0sfc6.weight.data = netm0s.fc6.weight.data
# net.m0sfc6.bias.data = netm0s.fc6.bias.data
# net.m0sfc7.weight.data = netm0s.fc7.weight.data
# net.m0sfc7.bias.data = netm0s.fc7.bias.data
# net.m0sfc8.weight.data = netm0s.fc8.weight.data
# net.m0sfc8.bias.data = netm0s.fc8.bias.data
# net.m0sfc9.weight.data = netm0s.fc9.weight.data
# net.m0sfc9.bias.data = netm0s.fc9.bias.data

# net.m0sbn3.weight.data = netm0s.bn3.weight.data
# net.m0sbn3.bias.data = netm0s.bn3.bias.data
# net.m0sbn4.weight.data = netm0s.bn4.weight.data
# net.m0sbn4.bias.data = netm0s.bn5.bias.data
# net.m0sbn5.weight.data = netm0s.bn5.weight.data
# net.m0sbn5.bias.data = netm0s.bn5.bias.data
# net.m0sbn6.weight.data = netm0s.bn6.weight.data
# net.m0sbn6.bias.data = netm0s.bn6.bias.data
# net.m0sbn7.weight.data = netm0s.bn7.weight.data
# net.m0sbn7.bias.data = netm0s.bn7.bias.data
# net.m0sbn8.weight.data = netm0s.bn8.weight.data
# net.m0sbn8.bias.data = netm0s.bnc8.bias.data

# net.m0sconv1.weight.data = netm0s.conv1.weight.data
# net.m0sconv1.bias.data = netm0s.conv1.bias.data
# net.m0sconv2.weight.data = netm0s.conv2.weight.data
# net.m0sconv2.bias.data = netm0s.conv2.bias.data

		  