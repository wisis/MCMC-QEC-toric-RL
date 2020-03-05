import faulthandler; faulthandler.enable()
import numpy as np
import random as rand
import copy
import collections 

from numba import jit, prange
from .toric_model import Toric_code
from .util import Action

rule_table = np.array(([[0, 1, 2, 3], [1, 0, 3, 2], 
                        [2, 3, 0, 1], [3, 2, 1, 0]]), dtype=int)    # Identity = 0
                                                                    # pauli_x = 1
                                                                    # pauli_y = 2
                                                                    # pauli_z = 3


class Chain:
    def __init__(self, size, p):
        self.toric = Toric_code(size)
        self.size = size
        self.p = p
        self.p_logical = 0
        self.flag = 0
        
        # lägg till nonzeros
            
    def update_chain(self):
        nbr_errors_added = 0
        if rand.random() < self.p_logical:
            new_matrix, nbr_errors_added = apply_random_logical(self.toric.qubit_matrix)
        else:
            new_matrix, nbr_errors_added = apply_random_stabilizer(self.toric.qubit_matrix)

        #r = r_chain(self.toric.qubit_matrix, new_matrix, self.p)
        
        

        #qubit_errors_current = np.count_nonzero(self.toric.qubit_matrix) or self.perevious_nonzero_count
        #qubit_errors_new = np.count_nonzero(new_matrix)

        #r = ((self.p / 3.0) / (1.0 - self.p)) ** (qubit_errors_new - qubit_errors_current)
        r = ((self.p / 3.0) / (1.0 - self.p)) ** (nbr_errors_added)
        

        if rand.random() < r:
            self.toric.qubit_matrix = new_matrix
        

    def plot(self, name):
        self.toric.syndrom('next_state')
        self.toric.plot_toric_code(self.toric.next_state, name)
#@profile
def parallel_tempering(init_toric, Nc=None, p=0.1, SEQ=2, TOPS=10, eps=0.1, steps=1000, iters=10, conv_criteria='error_based'):
    
    size = init_toric.system_size
    Nc = Nc or size
    # create the diffrent chains in an array
    # number of chains in ladder, must be odd
    try:
        Nc % 2 == 0
    except:
        print('Number of chains was not odd.')
    ladder = []  # ladder to store all chains
    p_end = 0.75  # p at top chain as per high-threshold paper
    tops0 = 0
    convergence_reached = 0
    nbr_errors_bottom_chain = []
    eq_count = np.zeros(16)
    eq_class_distr = []
    eq = []
    counter = 0
    nbr_steps_after_convergence = 100

    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')

    # add and copy state for all chains in ladder
    for i in range(Nc):
        p_i = p + ((p_end - p) / (Nc - 1)) * i
        ladder.append(Chain(size, p_i))
        ladder[i].toric = copy.deepcopy(init_toric)  # give all the same initial state
    ladder[Nc - 1].p_logical = 0.5  # set probability of application of logical operator in top chain

    bottom_equivalence_classes = np.zeros(steps, dtype=int)
    

    for j in range(steps):
        # run mcmc for each chain [steps] times
        for i in range(Nc):
            for _ in range(iters):
                ladder[i].update_chain()
        # now attempt flips from the top down
        for i in reversed(range(Nc - 1)):
            if i == (Nc - 2):
                ladder[i + 1].flag = 1
            if ladder[0].flag == 1:
                tops0 += 1
                ladder[0].flag = 0
            r_flip(ladder[i], ladder[i + 1])

        if conv_criteria == 'error_based':
            nbr_errors_bottom_chain.append(np.count_nonzero(ladder[0].toric.qubit_matrix))
            if tops0 >= TOPS:
                convergence_reached = conv_crit_error_based(ladder[0], nbr_errors_bottom_chain, eq_class_distr, tops0, TOPS, SEQ, eps)
        if conv_criteria == 'distr_based':
            if tops0 >= TOPS:
                convergence_reached = conv_crit_distr_based(ladder[0], eq, eq_count)
        if conv_criteria == 'majority_based':
                 if tops0 >= 1:
                         convergence_reached, majority_class = conv_crit_majority_based(ladder[0], eq, tops0, TOPS, SEQ) 
                         #returns the majority class that becomes obvious right when convergence is reached
        if convergence_reached and conv_criteria != 'majority_based':  # converged, append eq:s to list
            counter+=1
            eq_class_distr.append(define_equivalence_class(ladder[0].toric.qubit_matrix))
            if counter == nbr_steps_after_convergence: break 
        elif convergence_reached and conv_criteria == 'majority_based': 
                counter+=1
                eq_class_distr.append(define_equivalence_class(ladder[0].toric.qubit_matrix))
                #print("Majority class: ", majority_class)
                if counter == nbr_steps_after_convergence: 
                        break 
        
        bottom_equivalence_classes[j] = define_equivalence_class(ladder[0].toric.qubit_matrix)

    # plot all chains
    #for i in range(Nc):
    #    ladder[i].plot('Chain_' + str(i))

    # count number of occurrences of each equivalence class
    # equivalence_class_count[i] is the number of occurences of equivalence class number 'i'
    # if
    eq_class_count_BC = np.bincount(bottom_equivalence_classes, minlength=16)
    eq_class_count_AC = np.bincount(eq_class_distr, minlength=16)
    #print('After Count:\n',eq_class_count_AC)
    #print('Equivalence classes: \n', np.arange(16))
    #print('Before Count:\n', eq_class_count_BC)
    #print("NORM: ", np.linalg.norm(eq_class_count_AC))
    distr = np.divide(eq_class_count_AC, np.sum(eq_class_count_AC))
    return [distr, eq_class_count_BC,eq_class_count_AC,ladder[0]]

def conv_crit_error_based(bottom_chain, nbr_errors_bottom_chain, eq_class_distr, tops0, TOPS, SEQ, eps):#  Konvergenskriterium 1 i papper
    second_quarter = nbr_errors_bottom_chain[(len(nbr_errors_bottom_chain) // 4): (len(nbr_errors_bottom_chain) // 4) * 2]
    fourth_quarter = nbr_errors_bottom_chain[(len(nbr_errors_bottom_chain) // 4) * 3: (len(nbr_errors_bottom_chain) // 4) * 4]
    Average_second_quarter = sum(second_quarter) / (len(second_quarter))
    Average_fourth_quarter = sum(fourth_quarter) / (len(fourth_quarter))
    error = abs(Average_second_quarter - Average_fourth_quarter)
    if error > eps:
        tops0 = TOPS
    return tops0 == TOPS + SEQ  # true if converged

def conv_crit_distr_based(bottom_chain, eq, eq_count, norm_tol=2.5):
    eq_last = define_equivalence_class(bottom_chain.toric.qubit_matrix)
    eq = eq + [eq_last]
    eq_count[eq_last] += 1
    #bsump = np.sum(eq_count)
    Q2_count = np.zeros(16)
    Q4_count = np.zeros(16)

    l = len(eq)

    for i in range(l):
        if i >= l // 4 and i <= l // 2:
            Q2_count[eq[i]] = Q2_count[eq[i]] + 1
        if i >= (l * 3) // 4 and i < l:
            Q4_count[eq[i]] = Q4_count[eq[i]] + 1
    
    #for i in range(16):
        #print("ClassQ2: " + str(i))
    # print(Q2_count[i]/(np.sum(Q2_count)))
    #for i in range(16):
        #print("ClassQ4: " + str(i))
        #print(Q4_count[i]/(np.sum(Q4_count)))

    #print("Norm: " + str(np.linalg.norm(Q4_count - Q2_count)) )

    return (np.linalg.norm(Q4_count-Q2_count)) < norm_tol

def conv_crit_majority_based(bottom_chain, eq, tops0, TOPS, SEQ):
        count_last_quarter = None
        eq.append(define_equivalence_class(bottom_chain.toric.qubit_matrix))
        length = len(eq)
        if tops0 >= TOPS:
                count_second_half = collections.Counter(eq[length//2:])
                count_second_half = sorted(eq[length//2:], key=lambda x: -count_second_half[x])[0]
                count_last_quarter = collections.Counter(eq[(length-length//4):])
                count_last_quarter = sorted(eq[(length-length//4):], key=lambda x: -count_last_quarter[x])[0]
                if count_second_half-count_last_quarter == 0:
                        return tops0 >= SEQ+TOPS, count_last_quarter
                else: 
                    tops0 = TOPS        
                    return False, count_last_quarter
        return False, count_last_quarter

def r_flip(chain_lo, chain_hi):
    p_lo = chain_lo.p
    p_hi = chain_hi.p
    ne_lo = np.count_nonzero(chain_lo.toric.qubit_matrix)
    ne_hi = np.count_nonzero(chain_hi.toric.qubit_matrix)
    # compute eqn (5) in high threshold paper
    r = ((p_lo / p_hi) * ((1 - p_hi) / (1 - p_lo))) ** (ne_hi - ne_lo)
    if rand.random() < r:
        # flip them with prob r if r < 1
        temp = chain_lo.toric
        chain_lo.toric = chain_hi.toric
        chain_hi.toric = temp
        tempflag = chain_lo.flag
        chain_lo.flag = chain_hi.flag
        chain_hi.flag = tempflag


def apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[1]
    
    operator = int(rand.random() * 3) + 1  # operator to use, 2 (Y) will make both X and Z on the same layer
    orientation = int(rand.random() * 2)  # 0 - horizontal, 1 - vertical

    if orientation == 0:  # Horizontal
        if operator == 2:
            order = int(rand.random() * 2)  # make sure that we randomize which operator goes verically and horizontally
            temp_qubit_matrix, _ = apply_logical_horizontal(qubit_matrix, int(rand.random()*size), (order * 2 - 1) % 4)
            return apply_logical_horizontal(temp_qubit_matrix, int(rand.random()*size), (order * 2 + 1) % 4)
        else:
            return apply_logical_horizontal(qubit_matrix, int(rand.random()*size), operator)
    elif orientation == 1:  # Vertical
        if operator == 2:
            order = int(rand.random() * 2)  # make sure that we randomize which operator goes verically and horizontally
            temp_qubit_matrix, _ = apply_logical_vertical(qubit_matrix, int(rand.random()*size), (order * 2 - 1) % 4)
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

    return result_qubit_matrix, np.count_nonzero(new_operators)-np.count_nonzero(old_operators)


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

    return result_qubit_matrix, np.count_nonzero(new_operators)-np.count_nonzero(old_operators)


@jit(nopython=True)
def apply_stabilizer(qubit_matrix, row=int, col=int, operator=int):
        # gives the resulting qubit error matrix from applying (row, col, operator) stabilizer
        # doesn't update input qubit_matrix
        size = qubit_matrix.shape[1]
        nbr_errors_added = 0
        if operator == 1:
            qubit_matrix_layers = np.array([1, 1, 0, 0])
            rows = np.array([row, row, row, (row - 1) % size])
            cols = np.array([col, (col - 1) % size, col, col])
                
        elif operator == 3:
            qubit_matrix_layers = np.array([1, 0, 0, 1])
            rows = np.array([row, row, row, (row + 1) % size])
            cols = np.array([col, col, (col + 1) % size, col])
            
        result_qubit_matrix = np.copy(qubit_matrix)
                
        for i in range(4):
            value_before = result_qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]]
            result_qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]] = rule_table[operator][qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]]]
            value_after = result_qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]]
            if value_before == 0 and value_after > 0: nbr_errors_added+=1
            elif value_before > 0 and value_after == 0: nbr_errors_added = nbr_errors_added-1
        return result_qubit_matrix, int(nbr_errors_added)


def apply_random_stabilizer(qubit_matrix):
    # select random coordinates where to apply operator
    nbr_errors_added = 0
    size = qubit_matrix.shape[1]
    
    row = int(rand.random() * size)
    col = int(rand.random() * size)
    operator = int(rand.random() * 2)  # we only care about X and Z, and Y is represented by 2. Therefore:
    if operator == 0:
        operator = 3
    result_qubit_matrix, nbr_errors_added = apply_stabilizer(qubit_matrix, row, col, operator)
    
    return result_qubit_matrix, nbr_errors_added


def apply_stabilizers_uniform(qubit_matrix, p=0.5):
    size = qubit_matrix.shape[1]
    result_qubit_matrix = np.copy(qubit_matrix)
    random_stabilizers = np.random.rand(2, size, size)
    random_stabilizers = np.less(random_stabilizers, p) 
    
    # Numpy magic for iterating through matrix
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
    
