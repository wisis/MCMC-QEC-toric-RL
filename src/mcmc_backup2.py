import numpy as np
import random as rand
import copy
import collections 
from .toric_model import Toric_code
from .util import Action
from numba import jit 




rule_table = np.array(([[0, 1, 2, 3], [1, 0, 3, 2], 
                        [2, 3, 0, 1], [3, 2, 1, 0]]), dtype=int)    # Identity = 0
                                                                    # pauli_x = 1
                                                                    # pauli_y = 2
                                                                    # pauli_z = 3


def update_chain(qubit_matrix, p, cycles=int, allow_logical_flips = False, apply_stabilizers_in_start = True):
    p_logical = 1
    size = qubit_matrix.shape[1]
    new_matrix = copy.deepcopy(qubit_matrix)
    if apply_stabilizers_in_start == True: new_matrix  = apply_stabilizers_uniform(qubit_matrix)
    if rand.random() < p_logical:
    	if allow_logical_flips == True: new_matrix = apply_random_logical(qubit_matrix)
    new_matrix = apply_random_stabilizer(qubit_matrix)
    qubit_errors_current = np.count_nonzero(qubit_matrix)
    qubit_errors_new = np.count_nonzero(new_matrix)
    r = ((p / 3.0) / (1.0 - p)) ** (qubit_errors_new - qubit_errors_current)

    if rand.random() < r:
    	return new_matrix
    else:
    	return qubit_matrix

def error_count(qubit_matrix):
	vertical = qubit_matrix[0,:,:]
	horizontal = qubit_matrix[1,:,:]
	return np.count_nonzero(horizontal)+np.count_nonzero(vertical)


def parallel_tempering_mcmc(toric_model, p, Nc, cycles = int, swap_freq = int, seq = int, TOPS = int):
	delta = (0.75-p)/(Nc-1)
	list = [] #list of qubit_matrix that represents chains
	equivalence_class = [] #list of equivalence classes in first chain
	
	for i in range(Nc):
		list.append(copy.deepcopy(toric_model.qubit_matrix))
		list[i] = apply_stabilizers_uniform(list[i])
		#print(list[i])
	
	ladder = np.zeros(len(list)) # keeps track for every chain if it originates from top 
	ladder[-1] = 1  #top chain always originates in top chain 
	tops0 = 0 #indicator for number of unique samples that have propagated down 
	tops_counts = 0 #number of consecutive tops where convergence is fulfilled
	for j in range(Nc):
		if j<Nc-1:
			list[j] = update_chain(list[j], p+delta*j, 1, False, True)
		else:
			list[-1] = update_chain(list[-1], p+delta*j, 1, False, True)
	
	for i in range(cycles):
		
		if i%swap_freq != 0:
			
			for j in range(Nc):
				if j<Nc-1:
					list[j] = update_chain(list[j], p+delta*j, 1, False, False)
				else:
					list[-1] = update_chain(list[-1], p+delta*j, 1, True, False)
			
		elif i%swap_freq == 0:
			
			equiv, swaps = mcmc_swap(list, p, Nc)
			#print(ladder,swaps)	
			if tops0 > 0:
				equivalence_class.append(equiv) 
			
			ladder, swaps = overlap(ladder, swaps)
			
			length = len(equivalence_class)

			if tops0 >= TOPS:
				#print(tops0, ladder[0])
				#finding most frequent value in last half and last quarter
				count_second_half = collections.Counter(equivalence_class[length//2:])
				count_second_half = sorted(equivalence_class[length//2:], key=lambda x: -count_second_half[x])[0]
				count_last_quarter = collections.Counter(equivalence_class[(length-length//4):])
				count_last_quarter = sorted(equivalence_class[(length-length//4):], key=lambda x: -count_last_quarter[x])[0]
				if count_second_half-count_last_quarter == 0:
					if ladder[0] == 1:		
						ladder[0] = 0
						if count_second_half-count_last_quarter == 0:
							tops_counts = tops_counts+1
							if tops_counts>=seq:
								toric_model.qubit_matrix = list[0]
								print(i/swap_freq)
								return int(count_second_half)
								break 	
				else: 
					tops_counts = 0	
			
			if ladder[0] == 1:
				tops0 = tops0+1
				ladder[0] = 0
			
	toric_model.qubit_matrix = list[0]
	return "failure"

@jit(nopython=True)
def overlap(ladder, swaps):
	
	swaps = swaps[::-1] #shortcut for reversing array 
	ladder = ladder[::-1]
	
	#function that keeps track for every chain if it originates from top chain 
	for i in range(len(ladder)-1):
		if ladder[i] == 1 and swaps[i] == 1:
			ladder[i] = 0
			ladder[i+1] = 1
		ladder[0] = 1	
	swaps = np.zeros(len(ladder)) #swaps[::-1]
	ladder = ladder[::-1]
	
	return ladder, swaps


def mcmc_swap(list, p, Nc):
	#swaps all chains with tempering prob 
	list.reverse()
	delta = (0.75-p)/(Nc-1)
	swaps = np.zeros(len(list))#list that will hold swap or no swap for every chain. Later used to test convergence based on the number of unique samples that have propagated down 
	
	for i in range(len(list)-1):
		n_hi = error_count(list[i])
		n_lo = error_count(list[i+1])
		p_next = 0.75-delta*i
		p_current = 0.75-delta*(i+1)
		r = r_flip(p_current, p_next, n_lo, n_hi)
		
		if r > 1.0:
			list[i], list[i+1] = list[i+1], list[i]
			swaps[i] = 1
		else:
			prob = rand.random()
			if prob < r:
				list[i], list[i+1] = list[i+1], list[i]
				swaps[i] = 1
			else: 
				swaps[i] = 0
	list.reverse()
	
	swaps = swaps[::-1] #shortcut for reversing array 
	
	#print(swaps)
	return define_equivalence_class(list[0]), swaps
	

def plot(self, name):
    self.toric.syndrom('next_state')
    self.toric.plot_toric_code(self.toric.next_state, name)

@jit(nopython=True)
def r_flip(p, p_next, n_error=int, n_error_next=int):
	return (p/p_next*(1-p_next)/(1-p))**(n_error_next-n_error)

def apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[1]
    #operator = np.random.randint(1, 4)  # operator to use, 2 (Y) will make both X and Z on the same layer
    operator = int(1+rand.random()*3)
    #orientation = np.random.randint(0, 2)  # 0 - horizontal, 1 - vertical
    orientation = int(rand.random()*2)
    if orientation == 0:  # Horizontal
        if operator == 2:
            #order = np.random.randint(0, 2)  # make sure that we randomize which operator goes verically and horizontally
            order = int(rand.random()*2)
            temp_qubit_matrix = apply_logical_horizontal(qubit_matrix, int(rand.random()*size), (order * 2 - 1) % 4)
            return apply_logical_horizontal(temp_qubit_matrix, int(rand.random()*size), (order * 2 + 1) % 4)
        else:
            return apply_logical_horizontal(qubit_matrix, int(rand.random()*size), operator)
    elif orientation == 1:  # Vertical
        if operator == 2:
            #order = np.random.randint(0, 2)  # make sure that we randomize which operator goes verically and horizontally
            order = int(rand.random()*2)
            temp_qubit_matrix = apply_logical_vertical(qubit_matrix, int(rand.random()*size), (order * 2 - 1) % 4)
            return apply_logical_vertical(temp_qubit_matrix, int(rand.random()*size), (order * 2 + 1) % 4)
        else:
            return apply_logical_vertical(qubit_matrix, int(rand.random()*size), operator)


def apply_logical_vertical(qubit_matrix, col=int, operator=int):  # col goes from 0 to size-1, operator is either 1 or 3, corresponding to x and z
    size = qubit_matrix.shape[1]
    if operator == 1:  # makes sure the logical operator is applied on the correct layer, so that no syndromes are generated
        layer = 1
    else:
        layer = 0

    qubit_matrix_layers = np.full(size, layer, dtype=int)
    rows = np.arange(size)
    cols = np.full(size, col, dtype=int)

    old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
    new_operators = rule_table[operator][old_operators]

    result_qubit_matrix = qubit_matrix
    result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators

    return result_qubit_matrix


def apply_logical_horizontal(qubit_matrix, row=int, operator=int):  # col goes from 0 to size-1, operator is either 1 or 3, corresponding to x and z
    size = qubit_matrix.shape[1]
    if operator == 1:
        layer = 0
    else:
        layer = 1

    qubit_matrix_layers = np.full(size, layer, dtype=int)
    rows = np.full(size, row, dtype=int)
    cols = np.arange(size)

    old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
    new_operators = rule_table[operator][old_operators]

    result_qubit_matrix = qubit_matrix
    result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators

    return result_qubit_matrix


@jit(nopython=True)#, parallel=True)
def apply_stabilizer(qubit_matrix, row=int, col=int, operator=int):
    # gives the resulting qubit error matrix from applying (row, col, operator) stabilizer
    # doesn't update input qubit_matrix
    size = qubit_matrix.shape[1]
    if operator == 1: # 33.8% av tiden vs 54.8% av tiden jämfört med gamla, 62% av tiden med nya metoden
        
        qubit_matrix_layers = np.array([1, 1, 0, 0])
        rows = np.array([row, row, row, (row - 1) % size])
        cols = np.array([col, (col - 1) % size, col, col])
        
        '''
        # undviker att assigna massa saker och sparar på så sätt tid.
        result_qubit_matrix = np.copy(qubit_matrix)
        result_qubit_matrix[1, row, col] =              rule_table[1][qubit_matrix[1, row, col]]
        result_qubit_matrix[1, row, (col - 1) % size] = rule_table[1][qubit_matrix[1, row, (col - 1) % size]]
        result_qubit_matrix[0, row, col] =              rule_table[1][qubit_matrix[0, row, col]]
        result_qubit_matrix[0, (row - 1) % size, col] = rule_table[1][qubit_matrix[0, (row - 1) % size, col]]
        '''

    elif operator == 3: 
        qubit_matrix_layers = np.array([1, 0, 0, 1])
        rows = np.array([row, row, row, (row + 1) % size])
        cols = np.array([col, col, (col + 1) % size, col])

        '''
        old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
        new_operators = rule_table[operator][old_operators]
        result_qubit_matrix = np.copy(qubit_matrix)
        result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators
        '''

        '''
        # undviker att assigna massa saker och sparar på så sätt tid.
        result_qubit_matrix = np.copy(qubit_matrix)
        result_qubit_matrix[1, row, col] =              rule_table[3][qubit_matrix[1, row, col]]
        result_qubit_matrix[0, row, col] =              rule_table[3][qubit_matrix[0, row, col]]
        result_qubit_matrix[0, row, (col + 1) % size] = rule_table[3][qubit_matrix[0, row, (col + 1) % size]]
        result_qubit_matrix[1, (row - 1) % size, col] = rule_table[3][qubit_matrix[1, (row - 1) % size, col]]
        '''

    result_qubit_matrix = np.copy(qubit_matrix)

    #for i in prange(4):
    for i in range(4):
        result_qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]] = rule_table[operator][qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]]]

    return result_qubit_matrix
        

def apply_random_stabilizer(qubit_matrix):
    # select random coordinates where to apply operator
    size = qubit_matrix.shape[1]
    row = np.random.randint(0, size)  # gives int in [0, d-1]
    col = np.random.randint(0, size)
    operator = np.random.randint(0, 2)  # we only care about X and Z, and Y is represented by 2. Therefore:
    if operator == 0:
        operator = 3
    return apply_stabilizer(qubit_matrix, row, col, operator)


def apply_stabilizers_uniform(qubit_matrix):
    p = 0.5
    size = qubit_matrix.shape[1]
    result_qubit_matrix = np.copy(qubit_matrix)
    random_stabilizers = np.random.rand(2, size, size)
    random_stabilizers = np.less(random_stabilizers, p) 
    
    # Numpy magic for iterating through matrixapply_stabilizers_uniform
    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            op, row, col = it.multi_index
            if op == 0:
                op = 3
            result_qubit_matrix = apply_stabilizer(result_qubit_matrix, row, col, op)
        it.iternext()
    return result_qubit_matrix


def define_equivalence_class(qubit_matrix):
    #checks odd and even errors in each layer
    #gives a combination of four numbers corresponding to an equivalence class

    #checks odd or even x-errors in first layer
    x1prov = np.count_nonzero(qubit_matrix[0] == 1)

    #checks odd or even z-errors in first layer
    z1prov = np.count_nonzero(qubit_matrix[0] == 3)

    #checks odd or even y-erros in first layer and adds them to total number of x/z errors in first layer
    y1prov = np.count_nonzero(qubit_matrix[0] == 2)
    x1 = x1prov + y1prov
    z1 = z1prov + y1prov

    #checks odd or even x-errors in second layer
    x2prov = np.count_nonzero(qubit_matrix[1] == 1)

    #checks odd or even z-errors in second layer
    z2prov = np.count_nonzero(qubit_matrix[1] == 3)

    #checks odd or even y-erros in second layer and adds them to total number of x/z errors in second layer
    y2prov = np.count_nonzero(qubit_matrix[1] == 2)
    x2 = x2prov + y2prov
    z2 = z2prov + y2prov

    # stores whether there was an odd or even number of errors
    x1 = x1 % 2
    z1 = z1 % 2

    x2 = x2 % 2
    z2 = z2 % 2

    return x1 + z1 * 2 + x2 * 4 + z2 * 8
    