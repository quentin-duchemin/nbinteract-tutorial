import time
import scipy as sc
from scipy import io

t0 = time.time()

def load_urls():
	with open('urls.txt', encoding="ISO-8859-1") as f:
		for line in f:
			urls = line.strip().split(',')
	return(urls)

if __name__ = "main":
	from subprocess import run
	urls = load_urls()
	num = 1
	for url in urls:
		try:
			cmd = "wget --quiet -O data"+str(num)+".mat "+url
			run(cmd,shell=True)
		except:
			print(num)
		num += 1
		
	
	print(time.time()-t0)