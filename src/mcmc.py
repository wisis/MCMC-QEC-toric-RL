import numpy as np
import random as rand
import copy

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

    def update_chain(self):
        if rand.random() < self.p_logical:
            new_matrix = apply_random_logical(self.toric.qubit_matrix)
        else:
            new_matrix = apply_random_stabilizer(self.toric.qubit_matrix)

        #r = r_chain(self.toric.qubit_matrix, new_matrix, self.p)

        qubit_errors_current = np.count_nonzero(self.toric.qubit_matrix)
        qubit_errors_new = np.count_nonzero(new_matrix)

        r = ((self.p / 3.0) / (1.0 - self.p)) ** (qubit_errors_new - qubit_errors_current)

        if rand.random() < r:
            self.toric.qubit_matrix = new_matrix

    def plot(self, name):
        self.toric.syndrom('next_state')
        self.toric.plot_toric_code(self.toric.next_state, name)


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


def apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[1]
    operator = int(rand.random() * 3) + 1  # operator to use, 2 (Y) will make both X and Z on the same layer
    orientation = int(rand.random() * 2)  # 0 - horizontal, 1 - vertical

    if orientation == 0:  # Horizontal
        if operator == 2:
            order = int(rand.random() * 2)  # make sure that we randomize which operator goes verically and horizontally
            temp_qubit_matrix = apply_logical_horizontal(qubit_matrix, np.random.randint(size), (order * 2 - 1) % 4)
            return apply_logical_horizontal(temp_qubit_matrix, np.random.randint(size), (order * 2 + 1) % 4)
        else:
            return apply_logical_horizontal(qubit_matrix, np.random.randint(size), operator)
    elif orientation == 1:  # Vertical
        if operator == 2:
            order = int(rand.random() * 2)  # make sure that we randomize which operator goes verically and horizontally
            temp_qubit_matrix = apply_logical_vertical(qubit_matrix, np.random.randint(size), (order * 2 - 1) % 4)
            return apply_logical_vertical(temp_qubit_matrix, np.random.randint(size), (order * 2 + 1) % 4)
        else:
            return apply_logical_vertical(qubit_matrix, np.random.randint(size), operator)


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


def apply_stabilizer(qubit_matrix, row=int, col=int, operator=int):
    # gives the resulting qubit error matrix from applying (row, col, operator) stabilizer
    # doesn't update input qubit_matrix
    size = qubit_matrix.shape[1]
    if operator == 1: # 33.8% av tiden vs 54.8% av tiden jämfört med gamla, 62% av tiden med nya metoden
        '''
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

    elif operator == 3: 
        '''qubit_matrix_layers = np.array([1, 0, 0, 1])
        rows = np.array([row, row, row, (row + 1) % size])
        cols = np.array([col, col, (col + 1) % size, col])

        old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
        new_operators = rule_table[operator][old_operators]

        result_qubit_matrix = np.copy(qubit_matrix)
        result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators'''
        # undviker att assigna massa saker och sparar på så sätt tid.
        result_qubit_matrix = np.copy(qubit_matrix)
        result_qubit_matrix[1, row, col] =              rule_table[3][qubit_matrix[1, row, col]]
        result_qubit_matrix[0, row, col] =              rule_table[3][qubit_matrix[0, row, col]]
        result_qubit_matrix[0, row, (col + 1) % size] = rule_table[3][qubit_matrix[0, row, (col + 1) % size]]
        result_qubit_matrix[1, (row - 1) % size, col] = rule_table[3][qubit_matrix[1, (row - 1) % size, col]]
    return result_qubit_matrix


def apply_random_stabilizer(qubit_matrix):
    # select random coordinates where to apply operator
    size = qubit_matrix.shape[1]
    row = int(rand.random() * size)
    col = int(rand.random() * size)
    operator = int(rand.random() * 2)  # we only care about X and Z, and Y is represented by 2. Therefore:
    if operator == 0:
        operator = 3
    return apply_stabilizer(qubit_matrix, row, col, operator)


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
    