import torch
import sys
import pandas as pd
import time
import copy
import importlib
import scipy as sc
import os


class Projection():
	def __init__(self, start_by_projection, dimension_projection, initialization, normalization):
		self.start_by_projection  = start_by_projection
		self.initialization       = initialization
		self.normalization        = normalization
		self.dimension_projection = dimension_projection
		my_path                   = os.path.abspath(os.path.dirname(__file__))
		path                      = os.path.join(my_path, 'simulate_signal/eigenvectors.mat')
		self.eigenvectors         = sc.io.loadmat(path)['eigenvectors']

	def initialization_first_layer(self, net, device):
		if self.initialization == 'PCA':
			dimension_proj = net.fc1.out_features
			eigenvectors = torch.tensor(self.eigenvectors[:,:self.dimension_projection].T,dtype=torch.float,device=device)
			net.fc1.weight.data = eigenvectors
		elif self.initialization == 'Fixlayer':
			dimension_proj = net.fc1.out_feature
			self.eigenvectors = torch.tensor(self.eigenvectors[:,:self.dimension_projection], dtype=torch.float, device=device)
		return net

	def project(self, inputs):
		if self.initialization == 'Fixlayer':
			inputs = inputs.mm(self.eigenvectors)
		return inputs

	def dico_save(self):
		dic = {
				'start_by_projection'  : self.start_by_projection,
				'initialization'       : self.initialization,
 				'normalization'        : self.normalization,
 				'dimension_projection' : self.dimension_projection
            }
		return dic