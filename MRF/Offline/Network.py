from .Data_class import *
from ..BaseNetwork import *
from .Performances import *
from ..models import *
import time
import copy
import importlib
from torch import cuda  

class Network(BaseNetwork, Performances):
	""" 
	Class defining the whole neural network for training.
	The train method will use offline computed fingerprints.
	"""
	def __init__(self, name_model, loss, training_parameters, save_name, data_class, validation_settings, projection=None):
		""" New Network."""
		BaseNetwork.__init__(self, name_model, loss, training_parameters, data_class, projection=projection)
		self.save_name  = save_name
		Performances.__init__(self, validation_settings)
		
	def dico_save(self):
		""" Save the results and the settings of the training."""
		dic = Performances.dico_save(self)
		dic.update(BaseNetwork.dico_save(self))
		return dic
				
		
	def train(self, lr=0.001):	
		""" Launch the training using the parameter lr as learning rate."""
		if not os.path.exists('save_networks_offline'):
			os.mkdir('save_networks_offline')
		dtype = torch.float
		first_pass = True
		t0 = time.time()
	
		###################################### MODEL
		import importlib
		model = importlib.import_module('MRF.models.'+self.name_model)
		net = model.model(projection=self.projection, nb_params=len(self.trparas.params))
		net = net.to(self.device)
		if self.projection is not None:
			net = self.projection.initialization_first_layer(net, self.device)
		optimizer = optim.Adam(net.parameters(), lr=lr)
		########################################
		self.init_validation()
		
		for epoch in range(self.trparas.nb_epochs):
			loss_epoch = 0.0
			relative_error = np.zeros(len(self.trparas.params))
			grad = 0
			for i in range(self.num_files_validation+1,len(self.data_class.urls)+1):
				inputs_file, params_file = self.data_class.load_data(i)
				try:
					assert(inputs_file.shape[0] != 0)
				except:
					print(' Error reading file : ',self.data_class.urls[i])
				else:
					inputs_file = self.data_class.add_noise_batch(inputs_file)
					ndata = inputs_file.shape[0]
					k = 0
					while ((k+1)*self.trparas.batch_size <= ndata):
						# zero the parameter gradients
						net.train()
						optimizer.zero_grad()
						inputs = torch.tensor(inputs_file[k*self.trparas.batch_size:(k+1)*self.trparas.batch_size,:], dtype=dtype)
						params = torch.tensor(params_file[k*self.trparas.batch_size:(k+1)*self.trparas.batch_size,:], dtype=dtype)
						inputs = inputs.to(device=self.device)
						params = params.to(device=self.device)
						# forward + backward + optimize
						outputs = net(inputs)
						loss = self.loss_function(outputs, params, self.trparas.batch_size)
						loss.backward()
						optimizer.step()
						# tracking gradient norm, loss and relative error
						total_norm = 0
						for p in net.parameters():
							param_norm = p.grad.detach().norm(2)
							total_norm += param_norm.item() ** 2
							total_norm = total_norm ** (1. / 2)

						if first_pass:
							first_pass = False
							self.trparas.nb_iterations = (ndata* (len(self.data_class.urls)-self.num_files_validation)) // self.trparas.batch_size
						loss_epoch     += loss.detach().item() / self.trparas.nb_iterations
						relative_error += ((self.compute_relative_errors(outputs.detach(),params,self.trparas.batch_size)).cpu()).numpy() / self.trparas.nb_iterations
						grad           += total_norm / self.trparas.nb_iterations
						k += 1

			print('EPOCH',loss_epoch, ' time ',time.time()-t0)
			if self.validation:
				net.eval()
				with torch.no_grad():
					estimations_validation_graph = net(self.dico_validation) 
					estimations_validation = estimations_validation_graph.cpu().detach()
					self.validation_step(estimations_validation)

			self.losses.append(loss_epoch)
			self.training_relative_errors.append(relative_error)
			self.gradients.append(grad)
			
			# Saving the results of the training 
			dic = {
		      'NN': net.state_dict(),
 				'learning_rate' : lr,
 				'time_per_epoch' : (time.time() -t0) / (epoch + 1)
            }
			dic.update(self.dico_save())
			torch.save(dic, 'save_networks_offline/network_'+self.save_name)
			
		print('Training_Finished')
		print('Total_time',time.time()-t0)