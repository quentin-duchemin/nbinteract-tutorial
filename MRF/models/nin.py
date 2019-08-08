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
			self.fc1  = nn.Linear(666, self.projection.dimension_projection)
			self.fc2  = nn.Linear(self.projection.dimension_projection, 128)
			self.fc3  = nn.Linear(128, 512)
			self.fc4  = nn.Linear(512, 1024)
			self.cv5  = nn.Conv1d(1, 8, 5)
			self.cv6  = nn.Conv1d(8, 8, 5)
			self.av6  = nn.AvgPool1d(4)
			self.cv7  = nn.Conv1d(8, 8, 4)
			self.av7  = nn.AvgPool1d(4)
		
			self.cv8  = nn.Conv1d(1, 8, 5)
			self.cv9  = nn.Conv1d(8, 8, 5)
			self.av9  = nn.AvgPool1d(4)
			self.cv10 = nn.Conv1d(8, 4, 4)
			self.av10 = nn.AvgPool1d(4)
		
			self.fc11 = nn.Linear(116, 58)
			self.fc12 = nn.Linear(58, 29)
			self.fc13 = nn.Linear(29, self.nb_params)
		  
	def forward(self, s):
		if self.projection.initialization != 'Fixlayer':
			s = self.fc1(s)
		if self.projection.normalization:
			s = self.normalization_post_projection(s)
		s = F.relu(self.fc2(s))
		s = F.relu(self.fc3(s))
		s = F.relu(self.fc4(s))
		s = s.unsqueeze(1)
		s = F.relu(self.cv5(s))
		s = F.relu(self.av6(self.cv6(s)))
		s = F.relu(self.av7(self.cv7(s)))
		s = s.view(s.shape[0], -1)
		s = s.unsqueeze(1)
		s = F.relu(self.cv8(s))
		s = F.relu(self.av9(self.cv9(s)))
		s = F.relu(self.av10(self.cv10(s)))
		s = s.view(s.shape[0], -1)
		s = F.relu(self.fc11(s))
		s = F.relu(self.fc12(s))
		s = self.fc13(s)
		return s
		