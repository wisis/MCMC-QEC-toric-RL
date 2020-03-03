from .toric_model import Toric_code
from .mcmc import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time 
import multiprocessing as mp


#print("Number of processors: ", mp.cpu_count())

toric = Toric_code(5)
p_error = 0.2
toric.generate_random_error(p_error)

t1 = time.time()
success= 0
for i in range(5):
	initial = define_equivalence_class(toric.qubit_matrix)
	#print("actual class: ", define_equivalence_class(toric.qubit_matrix))
	result = parallel_tempering_mcmc(toric, p =p_error, Nc = toric.system_size, cycles = 1000000, swap_freq = 10, seq = 2, TOPS = 10)
	#print("predicted class: ", result)
	if initial == result:
		success = success + 1
		print("success")
	else:
		print("failure")
print("success rate: ", success/5)
print("runtime: ", time.time()-t1)
print("average runtime: ", (time.time()-t1)/5)
