import torch
import numpy as np


class Performances():
	"""
	Class designed to handle the computations and the definition of the validation loss and errors.
	"""
	def __init__(self, validation_settings):
		self.losses                           = []
		self.training_relative_errors         = []
		for key, value in validation_settings.items():
			setattr(self, key, value)
		self.gradients                        = []
			
	def dico_save(self):
		""" Save the parameters of the instance of the class Performances."""
		return (self.__dict__).copy()
	
	def loss_function(self,outputs,params,size):
		raise NotImplementedError("Must override loss_function.")
        
	def compute_relative_errors(self, estimations_validation, parameters, size):
		raise NotImplementedError("Must override compute_relative_errors.")
		
	def validation_step(self, estimations_validation):
		"""
		Compute the loss and the relative errors on the parameters on the validation dataset. The parameter 'estimation_validation' represents the estimation of the network for the parameters on the validation dataset.
		"""
		self.losses_validation.append((self.loss_function(estimations_validation,self.params_validation, self.validation_size)).cpu().detach().numpy())
		self.validation_relative_errors.append((self.compute_relative_errors(estimations_validation, self.params_validation, self.validation_size)).cpu().detach().numpy())

	def init_validation(self):
		"""
		Define the validation dataset.
		"""
		if self.validation:
			self.losses_validation = []
			self.validation_relative_errors = []
			self.dico_validation = np.zeros((self.validation_size,666))
			self.params_validation = np.zeros((self.validation_size,6))
			for i in range(self.validation_size):
				self.dico_validation[i,:], self.params_validation[i,:] = self.data_class.__getitem__(0)
			self.dico_validation = torch.tensor(self.dico_validation, dtype=torch.float, device='cpu')
			self.params_validation = torch.tensor(self.params_validation, dtype=torch.float, device='cpu')