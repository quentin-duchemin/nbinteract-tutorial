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
		self.losses_small_validation.append((self.loss_function(estimations_validation[:self.small_validation_size],self.params_validation[:self.small_validation_size], self.small_validation_size)).cpu().detach().numpy())
		self.small_validation_relative_errors.append((self.compute_relative_errors(estimations_validation[:self.small_validation_size], self.params_validation[:self.small_validation_size], self.small_validation_size)).cpu().detach().numpy())
		
	def init_validation(self):
		"""
		Define the validation dataset.
		"""
		if self.validation:
			self.losses_validation, self.validation_relative_errors, self.losses_small_validation, self.small_validation_relative_errors= [],[],[],[]
			dico_validation, params_validation = self.data_class.load_data(1)
			num_files = 1
			ndata = params_validation.shape[0]
			count = ndata
			while count < self.validation_size:
				num_files += 1
				inputs, parameters = self.data_class.load_data(num_files)
				dico_validation = np.concatenate((dico_validation, inputs), axis=0)
				params_validation = np.concatenate((params_validation, parameters), axis=0)
				count += ndata
			self.num_files_validation = num_files
			dico_validation = dico_validation[:self.validation_size,:]
			params_validation = params_validation[:self.validation_size,:]
			dico_validation = torch.tensor(self.data_class.add_noise_batch(dico_validation), dtype=torch.float)
			self.dico_validation = (dico_validation).to(device=self.device)
			self.params_validation = torch.tensor(params_validation, dtype=torch.float, device='cpu')
		else:
			self.num_files_validation = 0
