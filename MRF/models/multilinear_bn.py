import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from ..BaseModel import * 


class model(BaseModel):
	def __init__(self, nb_params=None, projection=None, ghost=False):
		super(model, self).__init__(True, False, nb_params=nb_params, projection=projection, ghost=ghost)
		if not self.ghost:
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
		
				
		
			self.fc3 = nn.Linear(30, 512)
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
			self.fc9 = nn.Linear(16, self.nb_params)
		

		
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

		s = F.relu(self.bn3(self.fc3(s)))
		s = F.relu(self.bn4(self.fc4(s)))
		s = F.relu(self.bn5(self.fc5(s)))
		s = F.relu(self.bn6(self.fc6(s)))
		s = F.relu(self.bn7(self.fc7(s)))
		s = F.relu(self.bn8(self.fc8(s)))
		s = self.fc9(s)
		return s