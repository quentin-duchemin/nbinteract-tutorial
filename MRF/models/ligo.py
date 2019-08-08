import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from ..BaseModel import * 

# architecture inspired from the article : 

# Deep Learning for real-time gravitational wave detection and parameter estimation: Results with Advanced LIGO data


class model(BaseModel):
	def __init__(self, nb_params=None, projection=None, ghost=False):
		super(model, self).__init__(True, True, nb_params=nb_params, projection=projection, ghost=ghost)
		if not self.ghost:
			self.assert_projection_defined()
			self.assert_joint_learning()
			self.fc1  = nn.Linear(666, self.projection.dimension_projection)
			self.fc2  = nn.Linear(self.projection.dimension_projection, 128)
			self.fc3  = nn.Linear(128, 512)
			self.fc4  = nn.Linear(512, 1024)
			self.cv5  = nn.Conv1d(1,64,4)
			self.avg5 = nn.AvgPool1d(2)
			self.cv6  = nn.Conv1d(64,64,4)
			self.avg6 = nn.AvgPool1d(2)
			self.cv7  = nn.Conv1d(64,128,4)
			self.avg7 = nn.AvgPool1d(4)
			self.cv8  = nn.Conv1d(128,64,4)
			self.avg8 = nn.AvgPool1d(4)
			
			self.fc9  = nn.Linear(896, 128)
			self.fc10 = nn.Linear(128, 64)
			self.fc11 = nn.Linear(64, 16)
			self.fc12 = nn.Linear(16, self.nb_params)
		  
	def forward(self, s):
		if self.projection.initialization != 'Fixlayer':
			s = self.fc1(s)
		if self.projection.normalization:
			s = self.normalization_post_projection(s)
		s = F.relu(self.fc2(s))
		s = F.relu(self.fc3(s))
		s = F.relu(self.fc4(s))
		s = s.unsqueeze(1)
		s = F.relu(self.avg5(self.cv5(s)))
		s = F.relu(self.avg6(self.cv6(s)))
		s = F.relu(self.avg7(self.cv7(s)))
		s = F.relu(self.avg8(self.cv8(s)))
		s = s.view(s.shape[0], -1)
		s = F.relu(self.fc9(s))
		s = F.relu(self.fc10(s))
		s = F.relu(self.fc11(s))
		s = self.fc12(s)
		return s
		
		
