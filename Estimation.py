from Data_class import *
import sys
import pandas as pd
import time
import copy
import importlib

sys.path.append("./models")

nametoparam = {'m0s':[0],'T1':[1],'T2':[2],'The three parameters':[0,1,2]}
paramtoname =  {0:'m0s',1:'T1',2:'T2'}

def scaling(a):
	return ((a-0.1)/6.)

def rescale(a):
	return (6*a+0.1)
    
class Estimation(Data_class):
	def __init__(self, name_model, noise_type='Standard', noise_level=1./30., sampling='Log', loss='MSE-Log', batch_size = 64, nb_iterations = 100, nb_epochs=5000, params = [1], initialization = 'Random', min_values = params_min_values, max_values = params_max_values, validation=True, save_samples=True, validation_size=1000, t2_wrt_t1='below_percent',save_name='1'):
		super(Estimation, self).__init__(batch_size, nb_iterations, noise_type, noise_level, sampling, min_values, max_values, t2_wrt_t1)
		self.params            = params
		self.loss              = loss
		self.initialization    = initialization
		self.name_model        = name_model
		self.nb_epochs         = nb_epochs
		self.validation        = validation
		self.save_samples      = save_samples
		self.validation_size   = validation_size
		self.dico_validation   = []
		self.params_validation = []
		self.samples           = []
		self.save_name         = save_name

	def loss_function(self,outputs,params):
		if len(self.params)==1:
			return  (outputs.reshape(-1)-self.transform(params[:,self.params].reshape(-1))).pow(2).sum() / self.batch_size
		else:
			return  (outputs[:,self.params]-self.transform(params[:,self.params])).pow(2).sum() / self.batch_size

	def compute_relative_errors(self, estimations_validation, parameters, size):
		if len(self.params)==1:
			return  ( torch.abs( self.transform(estimations_validation.reshape(-1))-parameters[:,self.params].reshape(-1)) /(parameters[:,self.params].reshape(-1)) ).sum(dim=0) / size
		else:
			return  ( torch.abs( self.transform(estimations_validation[:,self.params])-parameters[:,self.params]) /(parameters[:,self.params]) ).sum(dim=0) / size

	def transform(self, outputs):
		if self.loss =='MSE-Log':
			return (10**outputs)
		elif self.loss =='MSE':
			return outputs
		elif self.loss=='MSE-Inverse':
			return (1./outputs)
		elif self.loss=='MSE-Scaling':
			return rescale(outputs)

	def study_estimator(self, net, signals):
		model = importlib.import_module('models.'+net['name_model'])
		netw = model.model()
		device = torch.device('cpu')
		netw.load_state_dict(net['NN'])
		netw.eval()
		mean = 0
		variance = 0
		with torch.no_grad():
			out = self.eval_net(netw, signals)
			output = out.numpy()
			mean = np.mean(output, axis=0)
			std = np.std(output, axis=0)
		return std, mean

	def add_noise(self,fing):
	    fingerprint = copy.deepcopy(fing)
	    l = len(fingerprint)
	    np.random.seed()
	    if self.noise_type == 'SNR':
	        noise = np.random.normal(0, 1, l)
	        signal_Power = np.linalg.norm(fingerprint)
	        noise_Power = np.linalg.norm(noise)
	        cst = signal_Power / (noise_Power * self.noise_level)
	        noise = noise * cst
	    elif self.noise_type == 'Standard':
	        noise = np.random.normal(0, self.noise_level, l)
	    return fingerprint + noise
	  
	def CRB(self, y, weights, var=1.):
		I = np.dot(y[:,1:4].T,y[:,1:4]) / (var**2)
		Im1 = np.linalg.inv(I)
	    # Optimize for the average of all parameters
		C = np.sum(Im1.diagonal() * weights)
	    # Normalize the cost; the abs is just in case the inversion failed
		C = abs(C)
		return C

	def nlls(self, fingerprint, nbnoise=5):
		def simu(x,para0,para1,para2,para3,para5):
			y,_ = simulate_MT_ODE(self.x, TR, self.t, para0, para1, para2, para3, para1, para5)
			return y[:,0]
		signals = np.zeros((nbnoise,len(fingerprint)))
		paras = np.zeros((nbnoise,5))
		for i in range(nbnoise):
			signals[i,:] = self.add_noise(fingerprint)
		signals = torch.tensor(signals, dtype=torch.float, device='cpu')

		paras = np.zeros((nbnoise,5))
		bounds = ([0.1,0.1,0.01,10,1e-3],[0.5,6,3,100,1e-1])
		p0 = [0.5,1,0.1,40,1e-1]
		for i in range(nbnoise):
			optpara, _ = sc.optimize.curve_fit(simu,self.x,signals[i,:],p0=p0,bounds=bounds)
			paras[i,:] = np.array(optpara)
		stdnet, meannet = self.study_estimator(net, nbnoise, signals)
		return stdnet, meannet, np.std(paras, axis=0), np.mean(paras, axis=0)
		
	def eval_net(self, netw, signals):
		outputs = netw(signals)
		out = self.transform(outputs)
		return out
		
	def compute_CRBs(self, fingerprint, y):
		CRBs = np.zeros(3)
		if self.noise_type == 'SNR':
			normalize =  np.linalg.norm(fingerprint) / (self.noise_level * np.sqrt(len(fingeprint)))
			CRBs[0] = np.sqrt(self.CRB(y, [1, 0, 0])) * normalize
			CRBs[1] = np.sqrt(self.CRB(y, [0, 1, 0])) * normalize
			CRBs[2] = np.sqrt(self.CRB(y, [0, 0, 1])) * normalize
		elif self.noise_type == 'Standard':
			CRBs[0] = np.sqrt(self.CRB(y, [1, 0, 0], var=self.noise_level**2))
			CRBs[1] = np.sqrt(self.CRB(y, [0, 1, 0], var=self.noise_level**2))  
			CRBs[2] = np.sqrt(self.CRB(y, [0, 0, 1], var=self.noise_level**2)) 
		return CRBs

	def compute_NLLS(self, files,nb_noise=1):
		Z = np.zeros((12,12,12,nb_noise,5))
		for i in range(12): #m0s
			for j in range(12): #t1
				for l in range(12): #t2
					for k in range(nb_noise):
						Z[i,j,l,k,:] = files[k][i,j,l,:]
		return np.mean(Z,axis=3), np.std(Z,axis=3)
		
	def local_study(self, net, m0s, T1, T2, nNoise, indm0s=0, indt1=0, indt2=0, r=40, T2s=1e-2, Pd=1., NLLS=False):
		pos = [i+1 for i in range(len(self.params))]
		bars = ('CRB','NN', 'NLLS')
		y,_ = simulate_MT_ODE_with_grads(self.x, TR, self.t, m0s, T1, T2, r ,T1, T2s)
		fingerprint = y[:,0]
		pars = [m0s,T1,T2,r,T1,T2s]
		CRBs = self.compute_CRBs(fingerprint, y)

		if NLLS:
			stdnet, meannet, stdls, meanls, CRBs = self.nlls(fingerprint, nbnoise=nNoise)
		else:
			filesfull = []
			for i in range(nNoise):
			  filesfull.append( np.reshape(sc.io.loadmat('noise_files/noise'+str(i+1)+'.mat')['SAVE'],(12,12,12,5)) )
			lsm0s = np.linspace(0.1,0.5,12)
			lst1 = np.logspace(np.log10(0.1),np.log10(6),12)
			lst2 = np.logspace(np.log10(0.01),np.log10(3),12)
			
			meanls, stdls  = self.compute_NLLS(filesfull,nb_noise=nNoise) 
			meanls, stdls = meanls[indm0s,indt1,indt2,:], stdls[indm0s,indt1,indt2,:]
			signals = np.zeros((100,len(fingerprint)))
			for i in range(100):
				signals[i,:] = self.add_noise(fingerprint)
			signals = torch.tensor(signals, dtype=torch.float, device='cpu')
			stdnet, meannet = self.study_estimator(net, signals)

		plt.figure(figsize=(12,3))
		nbparams = len(self.params)
		for i in range(nbparams):
			plt.subplot(1,nbparams,i+1)
			plt.ylabel(paramtoname[self.params[i]])
			plt.errorbar(1, pars[self.params[i]], CRBs[i], fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, label='CRB')
			plt.errorbar(2, meannet[i], stdnet[i], fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=0, label='NN')
			plt.errorbar(3, meanls[i], stdls[i], fmt='o', color='green', ecolor='lightgray', elinewidth=3, capsize=0, label='NLLS')
			plt.xticks(pos, bars)
			print(paramtoname[self.params[i]]+' :',pars[self.params[i]])
			print('                      sqrt{CRB} : ', round(CRBs[i],4) )
			print('NEURAL NETWORK mean : ', round(meannet[i],4), '    std : ', round(stdnet[i],4), '    sqrt{CRB} / std   : ', round(CRBs[i]/np.float(stdnet[i]),4) )
			print('NON LINEAR LS  mean : ', round(meanls[i],4), '    std : ', round(stdls[i],4), '    sqrt{CRB} / std   : ',round( CRBs[i]/np.float(stdls[i]),4) )
			print('\n')
			print('\n')
		plt.tight_layout()
		plt.show()

	def train(self, lr=0.001):	
		dtype = torch.float
		USE_CUDA = True and torch.cuda.is_available();
		device = torch.device('cuda' if USE_CUDA else 'cpu')
		losses = []
		losses_validation = []
		validation_relative_errors = []
		training_relative_errors = []

		first_pass = True
		name_params = ['m0s','T1','T2']
		t0 = time.time()
		
		# Parameters
		params = {'batch_size': self.batch_size, 'shuffle': True,'num_workers': 8,	'pin_memory':True}
		
		# Generators
		training_generator = data.DataLoader(self, **params)

		# Validation set
		if self.validation:
			with torch.no_grad():
				self.dico_validation = np.zeros((self.validation_size,666))
				self.params_validation = np.zeros((self.validation_size,6))
				for i in range(self.validation_size):
					self.dico_validation[i,:], self.params_validation[i,:] = self.__getitem__(0)
				self.dico_validation = torch.tensor(self.dico_validation, dtype=dtype, device=device)
				self.params_validation = torch.tensor(self.params_validation, dtype=dtype, device=device)
		###################################### CHOOSE THE MODEL
		import importlib
		
		model = importlib.import_module('models.'+self.name_model)
		net = model.model()
		net = net.to(device)
		
		if self.initialization == 'PCA':
			dimension_proj = net.fc1.out_features
			eigenvectors = sc.io.loadmat('eigenvectors.mat')['eigenvectors']
			eigenvectors = torch.tensor(eigenvectors[:,:dimension_proj].T,dtype=torch.float,device=device)
			net.fc1.weight.data = eigenvectors
		elif self.initialization == 'Fixlayer':
			dimension_proj = net.fc1.out_feature
			eigenvectors = sc.io.loadmat('eigenvectors.mat')['eigenvectors']
			eigenvectors = torch.tensor(eigenvectors[:,:dimension_proj], dtype=torch.float, device=device)

		optimizer = optim.Adam(net.parameters(), lr=lr)
		
		########################################
		
		for epoch in range(self.nb_epochs):  # loop over the dataset multiple times
		
			loss_epoch = 0.0
			relative_error = np.zeros(len(self.params))
			for inputs, params in training_generator:
				# zero the parameter gradients
				net.train()
				optimizer.zero_grad()
				inputs, params = torch.tensor(inputs, dtype=dtype, device=device), torch.tensor(params, dtype=dtype, device=device)
				if self.initialization == 'Fixlayer':
					inputs = inputs.mm(eigenvectors)
				# forward + backward + optimize
				outputs = net(inputs)
				loss = self.loss_function(outputs,params)
				loss.backward()
				optimizer.step()

				if first_pass:
					first_pass = False
					if self.save_samples:
						self.samples = params
				else:
					if self.save_samples:
						self.samples = torch.cat((self.samples,params),dim=0)
	        # print statistics
				loss_epoch += loss.item() / self.nb_iterations
				relative_error += ((self.compute_relative_errors(outputs.data,params,self.batch_size)).cpu()).numpy() / self.nb_iterations

			print('EPOCH',loss_epoch)
			if self.validation:
				net.eval()
				with torch.no_grad():
					estimations_validation = net(self.dico_validation) 
					losses_validation.append((self.loss_function(estimations_validation,self.params_validation)).cpu().numpy())
					validation_relative_errors.append((self.compute_relative_errors(estimations_validation, self.params_validation, self.validation_size)).cpu().numpy())
					
			losses.append(loss_epoch)
			training_relative_errors.append(relative_error)

			if epoch % 10 == 0:
				torch.save({
					'name_model' : self.name_model,
		         'NN': net.state_dict(),
		         'optimizer': optimizer.state_dict(),
		         'training_loss': np.array(losses),
	 				'learning_rate' : lr,
	 				'loss' : self.loss,
	 				'sampling' : self.sampling,
	 				'initialization' : self.initialization,
	 				'batch_size' : self.batch_size,
	 				'nb_iterations' : self.nb_iterations,
	 				'nb_epochs' : self.nb_epochs,
	 				'noise_type' : self.noise_type,
	 				'noise_level' : self.noise_level,
	 				'params' : self.params,
	 				'min_values' : self.min_values,
	 				'max_values' : self.max_values,
	 				'validation' : self.validation,
	 				'dico_validation' : self.dico_validation,
	 				'params_validation': self.params_validation,
	 				'validation_loss' : np.array(losses_validation),
	 				'validation_relative_errors' : np.array(validation_relative_errors),
	 				'training_relative_errors' : np.array(training_relative_errors),
	 				'samples' : ((self.samples).cpu()).numpy(),
	 				't2_wrt_t1' : self.t2_wrt_t1,
	 				'time_per_epoch' : (time.time() -t0) / (epoch + 1)
	            },  'save_networks/network_'+self.save_name)
		
		print('Finished Training')
		print('Total_time',time.time()-t0)