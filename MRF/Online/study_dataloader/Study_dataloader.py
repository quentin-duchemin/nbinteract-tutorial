import sys
import copy
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import sys
sys.path.append("./simulate_signal")
from simulation import *
import os
TR = 4.5e-3
import time
from torch.utils import data
from torch import optim
import scipy.integrate

def load_data_6params():
  x = sc.io.loadmat('./simulate_signal/OCT_resutl_MT_100_3s_low_SAR.mat')['x']
  x[1,:] = x[1,:] * 1e-3 # convert ms to s
  return x
    

def loguniform(low=0.1, high=1, size=None):
    return 10**(np.random.uniform(low, high, size))
    
    
class Test_data_class(data.Dataset):
	def __init__(self,batch_size, nb_iterations):
		self.batch_size   = batch_size
		self.nb_iterations  = nb_iterations
		self.x = load_data_6params()
		self.t = TR * np.array([i for i in range(666)])  


		
	def __len__(self):
		return self.batch_size * self.nb_iterations
        
	def sample(self):

		m0s = random.uniform(0.1,0.5)
		t1 = loguniform(np.log10(0.1),np.log10(6))

		t2 = loguniform(np.log10(0.01),np.log10(3))

		r = random.uniform(10,100)
		t2s = loguniform(-3,-1)
		return(np.array([m0s,t1,t2,r,t1,t2s]))
		
        
	def add_noise(self,fingerprint):
		l = len(fingerprint)
		noise = np.random.normal(0, 1./30., l)
		fingerprint += noise
		return fingerprint
        
	def __getitem__(self,idx):
		prms = self.sample()
		s,ds = simulate_MT_ODE(self.x, TR, self.t, prms[0], prms[1], prms[2], prms[3], prms[4], prms[5])
		noisy_fingerprint = self.add_noise(s[:,0])
		noisy_fingerprint /= np.linalg.norm(noisy_fingerprint)
		return noisy_fingerprint, prms
    
class Study_dataloader:
	def add_noise(self,fing):
	    fingerprint = copy.deepcopy(fing)
	    l = len(fingerprint)
	    noise = np.random.normal(0, 1./30., l)
	    return fingerprint + noise

	def train(self):	
	
		time_epoch = np.zeros((4,3))
		########################################
		i = 0
		for nb_iterations in [50,100,200,400]:
			j = 0
			for batch in [64,128,256]:
	
				# Parameters
				params = {'batch_size': batch, 'shuffle': True,'num_workers': 8,	'pin_memory':True}
				
				dataloader = Test_data_class(batch,nb_iterations)	
			
				# Generators
				training_generator = data.DataLoader(dataloader, **params)
				random.seed(0)
				np.random.seed(0)
				t0 = time.time()
				for inputs, params in training_generator:
					a = 1
				
				time_epoch[i,j] = time.time()-t0
				j += 1
				np.save('time_epoch.npy',time_epoch)
			i += 1

	
