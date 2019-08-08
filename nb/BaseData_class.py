from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
import sys
import scipy as sc
#from simulation_with_grads import *
import os
TR = 4.5e-3
import time
from torch.utils import data
from torch import optim
import scipy.integrate
import copy

    
class BaseData_class():
	"""
	Children classes define the way the fingerprints are generated. Pro-processing tasks (adding noise or normalization) are also handled by this class.
	"""
	def __init__(self, training_parameters, noise_type, noise_level, minPD, maxPD):
		self.trparas     = training_parameters
		self.noise_type  = noise_type
		self.noise_level = noise_level
		self.minPD 		  = minPD
		self.maxPD       = maxPD
		self.x           = load_data_6params()
		self.t           = TR * np.array([i for i in range(666)])  
		
	def sample(self):
		raise NotImplementedError("This method sample has to be implemented in the class inheriting from BaseData.")
