import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data



from ..BaseModel import * 


class model(BaseModel):
	def __init__(self, nb_params=None, projection=None, ghost=False):
		super(model, self).__init__(True, True, nb_params=nb_params, projection=projection, ghost=ghost)
		if not self.ghost:
			self.fc1 = nn.Linear(666, self.projection.dimension_projection)
			self.fc2 = nn.Linear(self.projection.dimension_projection, 2048)
			self.bn2 = nn.BatchNorm1d(2048)
			self.fc3 = nn.Linear(2048, 1536)
			self.bn3 = nn.BatchNorm1d(1536)
			self.fc4 = nn.Linear(1536, 1024)
			self.bn4 = nn.BatchNorm1d(1024)
			self.fc5 = nn.Linear(1024, 512)
			self.bn5 = nn.BatchNorm1d(512)
			self.fc6 = nn.Linear(512, 256)
			self.bn6= nn.BatchNorm1d(256)
			self.fc7 = nn.Linear(256, 128)
			self.bn7 = nn.BatchNorm1d(128)
			self.fc8 = nn.Linear(128, 64)
			self.bn8 = nn.BatchNorm1d(64)
			self.fc9 = nn.Linear(64, 32)
			self.bn9 = nn.BatchNorm1d(32)
			self.fc10 = nn.Linear(32, 16)
			self.bn10= nn.BatchNorm1d(16)
			self.fc11= nn.Linear(16, self.nb_params)

        
	def forward(self, s):
		if self.projection.initialization != 'Fixlayer':
			s = self.fc1(s)
		s = F.relu(self.bn2(self.fc2(s)))
		s = F.relu(self.bn3(self.fc3(s)))
		s = F.relu(self.bn4(self.fc4(s)))
		s = F.relu(self.bn5(self.fc5(s)))
		s = F.relu(self.bn6(self.fc6(s)))
		s = F.relu(self.bn7(self.fc7(s)))
		s = F.relu(self.bn8(self.fc8(s)))
		s = F.relu(self.bn9(self.fc9(s)))
		s = F.relu(self.bn10(self.fc10(s)))
		s = self.fc11(s)   
		return s
        
        