from enum import Enum

nametoparam = {'m0s':[0],'T1':[1],'T2':[2],'The three parameters':[0,1,2], 'R':[3], 'T2s':[5], 'The five parameters':[0,1,2,3,5]}
paramtoname =  {0:'m0s',1:'T1',2:'T2',3:'R',5:'T2s'}

class Training_parameters():
	def __init__(self, batch_size, nb_iterations, nb_epochs, params, normalization):
		self.batch_size    = batch_size
		self.nb_iterations = nb_iterations
		self.nb_epochs     = nb_epochs
		self.params        = params
		self.normalization = normalization
		
	def dico_save(self):
		return self.__dict__	

class Enumerate(Enum):
	@classmethod
	def list(self):
		return [e.value for e_name, e in self.__members__.items()]
			
class Loss(Enumerate):
	MSELOG     = "MSE-Log"
	MSE        = "MSE"
	MSEInv     = "MSE-Inverse"
	MSEScaling = "MSE-Scaling"
	
class Normalization(Enumerate):
	WHITHOUT   = "Without"
	NOISYINPUT = "Noisy_input"
	AFTERPROJ  = "After_projection"

class NoiseType(Enumerate):
	SNR      = "SNR"
	STANDARD = "Standard"

class Initialization(Enumerate):
	PCA      = "PCA"
	RANDOM   = "Random"
	FIXLAYER = "Fixlayer"
	
class Sampling(Enumerate):
	LOG     = "Log"
	UNIFORM = "Uniform"
	
class T2wrtT1(Enumerate):
	NOCONSTRAINT = 'no_constraint'
	BELOW        = 'below'
	BELOWPERCENT = 'below_percent'
	
	
