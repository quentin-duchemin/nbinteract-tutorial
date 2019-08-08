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
			self.fc2 = nn.Linear(self.projection.dimension_projection, 1024)
			self.fc3 = nn.Linear(1024, 512)
			self.fc4 = nn.Linear(512, 256)
			self.fc5 = nn.Linear(256, 128)
			self.fc6 = nn.Linear(128, 64)
			self.fc7 = nn.Linear(64, 32)
			self.fc8 = nn.Linear(32, 16)
			self.fc9 = nn.Linear(16, 1)
		  
	def forward(self, s):
		if self.projection.initialization != 'Fixlayer':
			s = self.fc1(s)
		s = F.relu((self.fc2(s)))
		s = F.relu((self.fc3(s)))
		s = F.relu((self.fc4(s)))
		s = F.relu(self.fc5(s))
		s = F.relu((self.fc6(s)))
		s = F.relu((self.fc7(s)))
		s = F.relu((self.fc8(s)))
		s = self.fc9(s)
		return s