from ..BaseData_class import *
from ..simulate_signal import simulation 
    
class Data_class(data.Dataset, BaseData_class):
	"""
	Class defining the way the parameters space can be sampled.
	"""
	def __init__(self, training_parameters, noise_type, noise_level, minPD, maxPD, sampling, min_values, max_values, t2_wrt_t1):
		BaseData_class.__init__(self, training_parameters, noise_type, noise_level, minPD, maxPD)
		self.sampling      = sampling
		self.min_values    = min_values
		self.max_values    = max_values
		self.t2_wrt_t1     = t2_wrt_t1
		
	def dico_save(self):
		""" Save the parameters of the instance of the class Data_class."""
		dic = (self.__dict__).copy()
		del dic['trparas']
		dic.update(self.trparas.dico_save())
		return dic

	def __len__(self):
		"""
		Method required to use the Pytorch DataLoader. Return the number of fingerprints computed by epoch.
		"""
		return self.trparas.batch_size * self.trparas.nb_iterations
        
	def sample(self):
		"""
		Define the way the parameters space can be sampled.
		"""
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
       
	def __getitem__(self,idx):
		"""
		Method required to use the Pytorch DataLoader. Return a randomly drawn fingerprints with noise.
		"""
		prms = self.sample()
		s,ds = simulation.simulate_MT_ODE(self.x, TR, self.t, prms[0], prms[1], prms[2], prms[3], prms[4], prms[5])
		noisy_fingerprint = self.add_noise(s[:,0])
		return noisy_fingerprint, prms
