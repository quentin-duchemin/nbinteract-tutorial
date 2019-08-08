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
			self.bn2 = nn.BatchNorm1d(1024)
			self.fc3 = nn.Linear(1024, 768)
			self.bn3 = nn.BatchNorm1d(768)
			self.fc4 = nn.Linear(768, 640)
			self.bn4 = nn.BatchNorm1d(640)
			self.fc5 = nn.Linear(640, 512)
			self.bn5 = nn.BatchNorm1d(512)
			self.fc6 = nn.Linear(512, 384)
			self.bn6 = nn.BatchNorm1d(384)
			self.fc7 = nn.Linear(384, 256)
			self.bn7 = nn.BatchNorm1d(256)
			self.fc8 = nn.Linear(256, 192)
			self.bn8 = nn.BatchNorm1d(192)
			self.fc9 = nn.Linear(192, 128)
			self.bn9 = nn.BatchNorm1d(128)
			self.fc10 = nn.Linear(128, 64)
			self.bn10 = nn.BatchNorm1d(64)
			self.fc11 = nn.Linear(64, 32)
			self.bn11 = nn.BatchNorm1d(32)
			self.fc12 = nn.Linear(32, 16)
			self.bn12 = nn.BatchNorm1d(16)
			self.fc13 = nn.Linear(16, 8)
			self.bn13 = nn.BatchNorm1d(8)
			self.fc14 = nn.Linear(8, self.nb_params)
		  
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
		s = F.relu(self.bn11(self.fc11(s)))
		s = F.relu(self.bn12(self.fc12(s)))
		s = F.relu(self.bn13(self.fc13(s)))
		s = self.fc14(s)
		return s