from .Data_class import *
from .Performances import *
import time
import copy
import importlib
from ..BaseNetwork import *


def scaling(a):
	return ((a-0.1)/6.)

def rescale(a):
	return (6*a+0.1)
    
class Network(BaseNetwork, Performances):
	""" 
	Class defining the whole neural network for training.
	The train method will use online computed fingerprints.
	"""
	def __init__(self, name_model, loss, training_parameters, save_name, data_class, save_samples, validation_settings, projection=None):
		""" New Network."""
		BaseNetwork.__init__(self, name_model, loss, training_parameters, data_class, projection=projection)
		self.save_name    = save_name
		self.save_samples = save_samples
		Performances.__init__(self, validation_settings)
		
	def dico_save(self):
		""" Save the results and the settings of the training."""
		dic = Performances.dico_save(self)
		dic.update(BaseNetwork.dico_save(self))
		return dic
		
	def train(self, lr=0.001):	
		""" Launch the training using the parameter lr as learning rate."""
		if not os.path.exists('save_networks_online'):
			os.mkdir('save_networks_online')
		dtype = torch.float
		first_pass = True
		t0 = time.time()
		
		# Parameters
		params = {'batch_size': self.trparas.batch_size, 'shuffle': True,'num_workers': 8,	'pin_memory':True}
		
		# Generators
		training_generator = data.DataLoader(self.data_class, **params)

		###################################### MODEL
		import importlib
		model = importlib.import_module('MRF.models.'+self.name_model)
		net = model.model(projection=self.projection, nb_params=len(self.trparas.params))
		if self.projection is not None:
			net = self.projection.initialization_first_layer(net, self.device)
		net = net.to(self.device)
		optimizer = optim.Adam(net.parameters(), lr=lr)
		########################################
		self.init_validation()
		
		for epoch in range(self.trparas.nb_epochs):  # loop over the dataset multiple times
			loss_epoch = 0.0
			relative_error = np.zeros(len(self.trparas.params))
			grad = 0
			for inputs, params in training_generator:
				# zero the parameter gradients
				net.train()
				optimizer.zero_grad()
				inputs, params = torch.tensor(inputs, dtype=dtype, device=self.device), torch.tensor(params, dtype=dtype, device=self.device)
				inputs = self.projection.project(inputs)
				# forward + backward + optimize
				outputs = net(inputs)
				loss = self.loss_function(outputs, params, self.trparas.batch_size)
				loss.backward()
				optimizer.step()
				
				total_norm = 0
				for p in net.parameters():
					param_norm = p.grad.data.norm(2)
					total_norm += param_norm.item() ** 2
					total_norm = total_norm ** (1. / 2)

				if first_pass:
					first_pass = False
					if self.save_samples:
						self.samples = params
				else:
					if self.save_samples:
						self.samples = torch.cat((self.samples,params),dim=0)
				loss_epoch     += loss.detach().item() / self.trparas.nb_iterations
				relative_error += ((self.compute_relative_errors(outputs.detach(),params,self.trparas.batch_size)).cpu()).numpy() / self.trparas.nb_iterations
				grad           += total_norm / self.trparas.nb_iterations
				

			print('EPOCH',loss_epoch)
			if self.validation:
				net.eval()
				with torch.no_grad():
					estimations_validation_graph = net(self.dico_validation) 
					estimations_validation = estimations_validation_graph.cpu().detach()
					self.validation_step(estimations_validation)
					
			self.losses.append(loss_epoch)
			self.training_relative_errors.append(relative_error)
			self.gradients.append(grad)

			dic = {
		      'NN': net.state_dict(),
 				'learning_rate' : lr,
 				'time_per_epoch' : (time.time() -t0) / (epoch + 1)
            }
			dic.update(self.dico_save())
			net = net.to('cpu')
			torch.save(dic, 'save_networks_online/network_'+self.save_name)
			net = net.to(self.device)

		print('Finished Training')
		print('Total_time',time.time()-t0)
