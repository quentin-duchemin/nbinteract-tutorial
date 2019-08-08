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

params_min_values = np.array([0.1,0.01,0.1,10,1e-3])
params_max_values = np.array([0.5,6,3,100,1e-1])

def load_data_6params():
  my_path = os.path.abspath(os.path.dirname(__file__))
  path = os.path.join(my_path, "simulate_signal/OCT_resutl_MT_100_3s_low_SAR.mat")
  x = sc.io.loadmat(path)['x']
  x[1,:] = x[1,:] * 1e-3 # convert ms to s
  return x

def loguniform(low=0.1, high=1, size=None):
    return 10**(np.random.uniform(low, high, size))
    
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
	
	def dico_save(self):
		""" Save the parameters of the instance of the class BaseData_class."""
		dic = (self.__dict__).copy()
		del dic['trparas']
		dic.update(self.trparas.dico_save())
		return dic

	def add_noise(self,fing):
		"""
		Add noise to a given fingerprint and perform normalization if asked.
		"""
		fingerprint = copy.deepcopy(fing)
		l = len(fingerprint)
		np.random.seed()
		fingerprint = np.random.uniform(self.minPD,self.maxPD) * fingerprint
		if self.noise_type == 'SNR':
			noise = np.random.normal(0, 1, l)
			signal_Power = np.linalg.norm(fingerprint)
			noise_Power = np.linalg.norm(noise)
			cst = signal_Power / (noise_Power * self.noise_level)
			noise = noise * cst
		elif self.noise_type == 'Standard':
			noise = np.random.normal(0, self.noise_level, l)
		fingerprint += noise
		if self.trparas.normalization == 'Noisy_input':
			return fingerprint / np.linalg.norm(fingerprint)
		else:
			return fingerprint
	    	
	def add_noise_batch(self,fingerprints):
		"""
		Add noise to a given batch of fingerprints and perform normalization if asked.
		"""
		n,l = fingerprints.shape
		np.random.seed()
		fingerprints *= np.tile(np.random.uniform(self.minPD,self.maxPD,n).reshape(-1,1),(1,l))
		if self.noise_type == 'SNR':
			noise = np.random.normal(0, 1, (n,l))
			signal_Power = np.linalg.norm(fingerprints, axis=1)
			noise_Power = np.linalg.norm(noise,axis=1)
			cst = signal_Power / (noise_Power * self.noise_level)
			noise = noise * np.tile(cst.reshape(-1,1),(1,l))
		elif self.noise_type == 'Standard':
			noise = np.random.normal(0, self.noise_level, (n,l))
		fingerprints += noise
		if self.trparas.normalization == 'Noisy_input':
			return fingerprints / np.tile(np.linalg.norm(fingerprints,axis=1).reshape(-1,1), (1,l))
		else:
			return fingerprints