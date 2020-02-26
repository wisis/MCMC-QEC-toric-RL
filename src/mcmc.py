import numpy as np
import random as rand
import copy

from .toric_model import Toric_code
from .util import Action

rule_table = np.array(([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]), dtype=int)    # Identity = 0
                                                                                                # pauli_x = 1
                                                                                                # pauli_y = 2
                                                                                                # pauli_z = 3

class Chain:
    def __init__(self, size, p):
        self.toric = Toric_code(size)
        self.size = size
        self.p = p
        self.p_logical = 0
    
    def permute_error(self): # eventually rewrite to remove middle steps.
        self.toric.qubit_matrix = permute_error(self.toric.qubit_matrix, self.size, self.p, self.p_logical)

    def plot(self, name):
        self.toric.plot_toric_code(self.toric.next_state, name)

    def set_p_logical(self, p_logical):
        self.p_logical = p_logical

    def get_p(self):
        return self.p
    
    def get_qubit_matrix(self):
        return self.toric.qubit_matrix

    def get_toric(self):
        return copy.deepcopy(self.toric)

    def set_toric(self, new_toric):
        self.toric = copy.deepcopy(new_toric)

def r_flip(chain_lo, chain_hi): # flips always? ...
    p_lo = chain_lo.get_p()
    p_hi = chain_hi.get_p()

    ne_lo = np.count_nonzero(chain_lo.get_qubit_matrix())
    ne_hi = np.count_nonzero(chain_hi.get_qubit_matrix())
    r = ((p_lo / p_hi) * ((1 - p_hi) / (1 - p_lo))) ** (ne_hi - ne_lo)
    if r > 1:
        # flip them
        print('flipped')
        temp = chain_lo.get_toric()
        chain_lo.set_toric(chain_hi.get_toric())
        chain_hi.set_toric(temp)
        return r
    elif rand.random() < r:
            # flip them
            print('flipped')
            temp = chain_lo.get_toric()
            chain_lo.set_toric(chain_hi.get_toric())
            chain_hi.set_toric(temp)
            return r
    print('stayed')
    return r


'''
def apply_stabilizer(toric_model, row=int, col=int, operator=int):
    # operator is 1 (X <-> vertex) or 3 (Z <-> plaquette)
    d = toric_model.system_size #input np.array of form (qubit_matrix=int, row=int, col=int, add_operator=int)

    if operator == 1:
        action1 = Action(position = np.array([1, row, col]), action = operator)
        action2 = Action(position = np.array([1, row, (col-1)%d]), action = operator)
        action3 = Action(position = np.array([0, row, col]), action = operator)
        action4 = Action(position = np.array([0, (row-1)%d, col]), action = operator)
    elif operator == 3:
        action1 = Action(position = np.array([1, row, col]), action = operator)
        action2 = Action(position = np.array([0, row, col]), action = operator)
        action3 = Action(position = np.array([0, row, (col+1)%d]), action = operator)
        action4 = Action(position = np.array([1, (row+1)%(d), col]), action = operator)

    toric_model.step(action1)
    toric_model.step(action2)
    toric_model.step(action3)
    toric_model.step(action4)
'''


def apply_random_logical(qubit_matrix, size=int):
    operator = np.random.randint(1, 4)  # operator to use, 2 (Y) will make both X and Z on the same layer
    orientation = np.random.randint(0, 2)  # 0 - horizontal, 1 - vertical

    if orientation == 0:  # Horizontal
        if operator == 2:
            order = np.random.randint(0, 2)  # make sure that we randomize which operator goes verically and horizontally
            temp_qubit_matrix = apply_logical_horizontal(qubit_matrix, size, np.random.randint(size), (order * 2 - 1) % 4)
            return apply_logical_horizontal(temp_qubit_matrix, size, np.random.randint(size), (order * 2 + 1) % 4)
        else:
            return apply_logical_horizontal(qubit_matrix, size, np.random.randint(size), operator)
    elif orientation == 1:  # Vertical
        if operator == 2:
            order = np.random.randint(0, 2)  # make sure that we randomize which operator goes verically and horizontally
            temp_qubit_matrix = apply_logical_vertical(qubit_matrix, size, np.random.randint(size), (order * 2 - 1) % 4)
            return apply_logical_vertical(temp_qubit_matrix, size, np.random.randint(size), (order * 2 + 1) % 4)
        else:
            return apply_logical_vertical(qubit_matrix, size, np.random.randint(size), operator)


def apply_logical_vertical(qubit_matrix, size=int, col=int, operator=int):  # col goes from 0 to size-1, operator is either 1 or 3, corresponding to x and z
    if operator == 1:  # makes sure the logical operator is applied on the correct layer, so that no syndromes are generated
        layer = 1
    else:
        layer = 0

    qubit_matrix_layers = np.full(size, layer, dtype=int)
    rows = np.arange(size)
    cols = np.full(size, col, dtype=int)

    old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
    new_operators = rule_table[operator][old_operators]

    result_qubit_matrix = np.copy(qubit_matrix)
    result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators

    return result_qubit_matrix


def apply_logical_horizontal(qubit_matrix, size=int, row=int, operator=int):  # col goes from 0 to size-1, operator is either 1 or 3, corresponding to x and z
    if operator == 1:
        layer = 0
    else:
        layer = 1

    qubit_matrix_layers = np.full(size, layer, dtype=int)
    rows = np.full(size, row, dtype=int)
    cols = np.arange(size)

    old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
    new_operators = rule_table[operator][old_operators]

    result_qubit_matrix = np.copy(qubit_matrix)
    result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators

    return result_qubit_matrix


def test_apply_stabilizer(qubit_matrix, size=int, row=int, col=int, operator=int):
    # gives the resulting qubit error matrix from applying (row, col, operator) stabilizer
    # doesn't update input qubit_matrix
    if operator == 1:
        qubit_matrix_layers = np.array([1, 1, 0, 0])
        rows = np.array([row, row, row, (row - 1) % size])
        cols = np.array([col, (col - 1) % size, col, col])

    elif operator == 3:
        qubit_matrix_layers = np.array([1, 0, 0, 1])
        rows = np.array([row, row, row, (row + 1) % size])
        cols = np.array([col, col, (col + 1) % size, col])

    old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
    new_operators = rule_table[operator][old_operators]

    result_qubit_matrix = np.copy(qubit_matrix)
    result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators

    return result_qubit_matrix


def test_apply_random_stabilizer(qubit_matrix, size):
    # select random coordinates where to apply operator
    row = np.random.randint(0, size)  # gives int in [0, d-1]
    col = np.random.randint(0, size)
    operator = np.random.randint(0, 2)  # we only care about X and Z, and Y is represented by 2. Therefore:
    if operator == 0:
        operator = 3
    return test_apply_stabilizer(qubit_matrix, size, row, col, operator)


def apply_random_stabilizer(toric):
    toric.qubit_matrix = test_apply_random_stabilizer(toric.qubit_matrix, toric.system_size)


def apply_n_independent_random_stabilizers(toric, n=int):
    for i in range(0, n):
        apply_random_stabilizer(toric)


def apply_n_distinct_random_stabilizers(toric, n=int):
    stabilizer_array = np.zeros(2 * toric.system_size**2)
    stabilizer_array[:n] = np.ones(n, dtype=bool)
    np.random.shuffle(stabilizer_array)

    stabilizer_matrix = stabilizer_array.reshape(toric.system_size, toric.system_size, 2)
    for index, stab in np.ndenumerate(stabilizer_matrix):
        if stab:
            operator = index[2]
            if operator == 0:
                operator = 3
            toric.qubit_matrix = test_apply_stabilizer(toric.qubit_matrix, toric.system_size, index[0], index[1], operator)


def error_ratio(qubit_matrix_current, qubit_matrix_next, p=float):
    qubit_errors_current = np.count_nonzero(qubit_matrix_current)
    qubit_errors_new = np.count_nonzero(qubit_matrix_next)

    ratio = ((p / 3.0) / (1.0 - p)) ** (qubit_errors_new - qubit_errors_current)
    return ratio


def permute_error(qubit_matrix, size, p, p_logical):
    if np.random.rand() < p_logical:
        new_matrix = apply_random_logical(qubit_matrix, size)
    else:
        new_matrix = test_apply_random_stabilizer(qubit_matrix, size)

    r = error_ratio(qubit_matrix, new_matrix, p)
    if np.random.rand() < r:
        return new_matrix
    else:
        return qubit_matrix


def init_error(toric, qubit_matrix):
    toric.qubit_matrix = np.copy(qubit_matrix)
    toric.syndrom('next_state')
