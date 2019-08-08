import sys
import time
from ..BaseData_class import *

import scipy as sc
class Data_class(data.Dataset, BaseData_class):
	"""
	Class allowing to deal with the importation of the precomputed fingerprints.
	"""
	def __init__(self, training_parameters, noise_type, noise_level, minPD, maxPD, urls_file, ghost=False):
		"""
		New Data_class.
		"""
		BaseData_class.__init__(self, training_parameters, noise_type, noise_level, minPD, maxPD, ghost=ghost)
		if not ghost:
			self.urls_file = urls_file
			self.urls = self.load_urls()			
		
	def load_urls(self):
		"""
		Transform the text file containing the urls associated to the precomputed fingerprints.
		"""
		my_path = os.path.abspath(os.path.dirname(__file__))
		path = os.path.join(my_path, 'loading_data/'+self.urls_file)
		with open(path, encoding="ISO-8859-1") as f:
			for line in f:
				urls = line.strip().split(',')
		return(urls)
		
	def load_data_from_web(self,num):
		"""
		Load the file number 'num' containing precomputed fingerprints from the web.
		"""
		cmd = "wget --quiet -O data.mat "+self.urls[num]
		run(cmd,shell=True)
		data = sc.io.loadmat('data.mat')
		return data['s'].T, np.concatenate((data['m0s'],data['T1'],data['T2f'],data['R'],data['T2s']), axis=1)

	def load_data(self,num):
		"""
		Load the file number 'num' containing precomputed fingerprints previously saved in the folder 'loading_data'.
		"""
		my_path = os.path.abspath(os.path.dirname(__file__))
		path = os.path.join(my_path, '../../../new_master_thesis/offline_built_fingerprints/loading_data/data'+str(num)+'.mat')
		data = sc.io.loadmat(path) #'loading_data/data'+str(num)+'.mat') 
		return data['s'].T, np.concatenate((data['m0s'],data['T1'],data['T2f'],data['R'],data['T2s']), axis=1)
		
		
	def compute_fingerprints(self):
		t0 = time.time()
		my_path = os.path.abspath(os.path.dirname(__file__))
		path = os.path.join(my_path, 'loading_data/')
		for i in range(1,401):
			print('file ',i,' over 400. Time ',time.time()-t0)
			for k in range(2**16):
				if i==1 and k ==0:
					params = self.sample()
					s,ds = simulation.simulate_MT_ODE(self.x, TR, self.t, params[0], params[1], params[2], params[3], params[4], params[5])
					dico = s[:,0].reshape(1,-1)
					params = params.reshape(1,-1)
				else:
					prms = self.sample()
					s,ds = simulation.simulate_MT_ODE(self.x, TR, self.t, prms[0], prms[1], prms[2], prms[3], prms[4], prms[5])
					params = np.concatenate((params,prms.reshape(1,-1)), axis=1)
					dico = np.concatenate((dico,s[:,0].reshape(1,-1)), axis=1)
			np.save(os.path.join(path,'fingerprints'+str(i)+'.npy'),dico)
			np.save(os.path.join(path,'params'+str(i)+'.npy'),params)
			
					
				

	def sample(self):
		"""
		Define the sampling strategy used to built the precomputed fingerprints files. This method is only informative and will not be used in this offline framework.
		"""
		random.seed()
		np.random.seed()
		m0s = random.uniform(0,0.7)
		t1 = 2.8 * random.uniform(0,1) + 0.2
		t2f = t1 * ( random.uniform(0,1) * 0.5 + 0.005 )
		r = 490 * random.uniform(0,1) + 10
		t2s= 0.2 * 10**(-3) + random.uniform(0,1) * 150 * 10**(-3)
		return(np.array([m0s,t1,t2f,r,t1,t2s]))