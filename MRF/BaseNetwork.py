import torch
import pandas as pd
import time
import copy
import importlib


class BaseNetwork():
	"""
	Mother of Network classes that would contains the method for training.
	"""
	def __init__(self, name_model, loss, training_parameters, data_class, projection=None):
		self.data_class = data_class
		self.name_model = name_model
		self.loss       = loss
		self.trparas    = training_parameters
		self.projection = projection
		USE_CUDA = True and torch.cuda.is_available();
		self.device = torch.device('cuda' if USE_CUDA else 'cpu')
		
	def dico_save(self):
		""" Save the parameters of the instance of the class BaseNetwork."""
		dic = self.projection.dico_save()
		dic.update({'name_model': self.name_model, 'loss': self.loss})
		dic.update(self.data_class.dico_save())
		return dic

	def loss_function(self,outputs,params,size):
		"""
		Compute the loss function. 'outputs' is the output given by of the neural network and 'params' is the ground truth. 
		"""
		if len(self.trparas.params)==1:
			return  (outputs.reshape(-1)-self.transform_inv(params[:,self.trparas.params].reshape(-1))).pow(2).sum() / size
		else:
			return  (outputs[:,self.trparas.params]-self.transform_inv(params[:,self.trparas.params])).pow(2).sum() / size

	def compute_relative_errors(self, estimations, parameters, size):
		"""
		Compute the relative errors of the different parameters. 'estimations' is the outputs of the neural network and 'parameters' are the ground truth parameters.
		"""
		if len(self.trparas.params)==1:
			return  ( torch.abs( self.transform(estimations.reshape(-1))-parameters[:,self.trparas.params].reshape(-1)) /(parameters[:,self.trparas.params].reshape(-1)) ).sum(dim=0) / size
		else:
			return  ( torch.abs( self.transform(estimations[:,self.trparas.params])-parameters[:,self.trparas.params]) /(parameters[:,self.trparas.params]) ).sum(dim=0) / size

	def transform(self, outputs):
		"""
		Go from the output of the network to the estimated parameters.
		"""
		if self.loss == 'MSE-Log':
			return (10**outputs)
		elif self.loss == 'MSE':
			return outputs
		elif self.loss == 'MSE-Inverse':
			return (1./outputs)
		elif self.loss == 'MSE-Scaling':
			return rescale(outputs)
			
	def transform_inv(self, params):
		"""
		Go from the parameters to the target values defined by the loss type chosen.
		"""
		if self.loss == 'MSE-Log':
			return (torch.log10(params))
		elif self.loss == 'MSE':
			return params
		elif self.loss == 'MSE-Inverse':
			return (1./params)
		elif self.loss == 'MSE-Scaling':
			return scale(params)

	def study_estimator(self, net, signals):
		"""
		Return the mean and the standard deviation of the estimated parameters given from the neural network 'net' using the fingerprints given by 'signals'. 
		'signals' is built concatening noisy versions obtained from a same fingerprint.
		"""
		model = importlib.import_module('models.'+net['name_model'])
		netw = model.model()
		device = torch.device('cpu')
		netw.load_state_dict(net['NN'])
		netw.eval()
		with torch.no_grad():
			out = self.eval_net(netw, signals)
			output = out.numpy()
			mean = np.mean(output, axis=0)
			std = np.std(output, axis=0)
		return std, mean
	  
	def CRB(self, y, weights, var=1.):
		"""
		Compute the Cramer Rao Bound for the parameters specified by the vector 'weights'.
		"""
		I = np.dot(y[:,1:4].T,y[:,1:4]) / (var**2)
		Im1 = np.linalg.inv(I)
	    # Optimize for the average of all parameters
		C = np.sum(Im1.diagonal() * weights)
	    # Normalize the cost; the abs is just in case the inversion failed
		C = abs(C)
		return C

	def eval_net(self, netw, signals):
		"""
		Return the estimated parameters using the network 'netw' on the input batch of fingerprints 'signals'.
		"""
		outputs = netw(signals)
		out = self.transform(outputs)
		return out 