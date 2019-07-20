from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
import sys
sys.path.append("./simulate_signal")
from simulation import *
from simulation_with_grads import *
import os
TR = 4.5e-3
import time
from torch.utils import data
from torch import optim
import scipy.integrate


params_min_values = np.array([0.1,0.01,0.1,10,1e-3])
params_max_values = np.array([0.5,6,3,100,1e-1])

def load_data_6params():
  x = sc.io.loadmat('./simulate_signal/OCT_resutl_MT_100_3s_low_SAR.mat')['x']
  x[1,:] = x[1,:] * 1e-3 # convert ms to s
  return x

def loguniform(low=0.1, high=1, size=None):
    return 10**(np.random.uniform(low, high, size))
    
    
class Data_class(data.Dataset):
	def __init__(self,batch_size, nb_iterations, noise_type, noise_level, sampling, min_values, max_values, t2_wrt_t1):
		self.batch_size   = batch_size
		self.nb_iterations  = nb_iterations
		self.noise_type   = noise_type
		self.noise_level  = noise_level
		self.sampling     = sampling
		self.min_values   = min_values
		self.max_values   = max_values
		self.x = load_data_6params()
		self.t = TR * np.array([i for i in range(666)])  
		self.t2_wrt_t1  = t2_wrt_t1

		
	def __len__(self):
		return self.batch_size * self.nb_iterations
        
	def sample(self):
		random.seed()
		np.random.seed()
		if self.sampling == 'Log':
			m0s = random.uniform(self.min_values[0],self.max_values[0])
			t1 = loguniform(np.log10(self.min_values[1]),np.log10(self.max_values[1]))
			if self.t2_wrt_t1 == 'no_constraint':
				t2 = loguniform(np.log10(self.min_values[2]),np.log10(self.max_values[2]))
			elif self.t2_wrt_t1 == 'below':
				maxi = np.min([self.max_values[2],t1])
				if maxi <= self.min_values[2]:
					t2 = t1
				else:
					t2 = loguniform(np.log10(self.min_values[2]),np.log10(maxi))
			elif self.t2_wrt_t1 == 'below_percent':
				mini = np.max([self.min_values[2],0.005*t1])
				maxi = np.min([self.max_values[2],t1])
				if maxi <= mini:
					t2 = t1
				else:
					t2 = loguniform(np.log10(mini),np.log10(maxi))
			r = random.uniform(self.min_values[3],self.max_values[3])
			t2s = loguniform(np.log10(self.min_values[4]),np.log10(self.max_values[4]))
			return(np.array([m0s,t1,t2,r,t1,t2s]))
		
		elif self.sampling == 'Uniform':
			m0s = random.uniform(self.min_values[0],self.max_values[0])
			t1 = random.uniform(self.min_values[1],self.max_values[1])
			if self.t2_wrt_t1 == 'no_constraint':
				t2 = random.uniform(self.min_values[2],self.max_values[2])
			elif self.t2_wrt_t1 == 'below':
				maxi = np.min([self.max_values[2],t1])
				if maxi <= self.min_values[2]:
					t2 = t1
				else:
					t2 = random.uniform(self.min_values[2],maxi)
			elif self.t2_wrt_t1 == 'below_percent':
				mini = np.max([self.min_values[2],0.005*t1])
				maxi = np.min([self.max_values[2],t1])
				if maxi <= mini:
					t2 = t1
				else:
					t2 = random.uniform(mini,maxi)			
			r = random.uniform(self.min_values[3],self.max_values[3])
			t2s = loguniform(np.log10(self.min_values[4]),np.log10(self.max_values[4]))
			return(np.array([m0s,t1,t2,r,t1,t2s]))
        
	def add_noise(self,fingerprint):
		np.random.seed()
		if self.noise_type == 'SNR':
			l = len(fingerprintcp)
			noise = np.random.normal(0, 1, l)
			signal_Power = np.linalg.norm(fingerprint)
			noise_Power = np.linalg.norm(noise)
			cst = signal_Power / (noise_Power * self.noise_level)
			fingerprint += noise * cst
			return fingerprint
				
		elif self.noise_type == 'Standard':
			l = len(fingerprint)
			noise = np.random.normal(0, self.noise_level, l)
			fingerprint += noise
			return fingerprint
        
	def __getitem__(self,idx):
		prms = self.sample()
		s,ds = simulate_MT_ODE(self.x, TR, self.t, prms[0], prms[1], prms[2], prms[3], prms[4], prms[5])
		noisy_fingerprint = self.add_noise(s[:,0])
		noisy_fingerprint /= np.linalg.norm(noisy_fingerprint)
		return noisy_fingerprint, prms