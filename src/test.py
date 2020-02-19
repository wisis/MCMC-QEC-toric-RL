from .toric_model import Toric_code
from .mcmc import *

def main():
    size = 3
    toric = Toric_code(size)

    init_errors = np.array([[1, 0, 2, 1], [1, 1, 1, 1], [1, 2, 2, 1]])
    for arr in init_errors:
        print(arr)
        action = Action(position = arr[:3], action = arr[3])
        toric.step(action)
    '''
    action = Action(position = np.array([0, 2, 3]), action = 1)
    toric.step(action)
    action = Action(position = np.array([1, 3, 2]), action = 1)
    toric.step(action)
    action = Action(position = np.array([0, 3, 2]), action = 1)
    toric.step(action)
    action = Action(position = np.array([1, 2, 3]), action = 1)
    toric.step(action)
    '''

    toric.plot_toric_code(toric.next_state, 'hej')

    apply_n_independent_random_stabilizers(toric, 1000000)

    qubit_matrix = np.copy(toric.qubit_matrix)
    p = 0.1

    #for i in range(100):
    #    qubit_matrix = permute_error(qubit_matrix, size, p)

    res = input('plot? ')
    while not res == 'n':
        try:
            n = int(res)
        except:
            n = 1
        for i in range(n):
            qubit_matrix = permute_error(qubit_matrix, size, p)
            toric.qubit_matrix = np.copy(qubit_matrix)
        toric.plot_toric_code(toric.next_state, 'hej')
        res = input('plot? ')
    #print("vertex: " + str(toric.next_state[0, :, :]))
    #print("plaquette: " + str(toric.next_state[1, :, :]))
    

'''
    rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)
    opps = np.array([0, 1, 2])
    rules = rule_table[1][opps]
    print(rules)
'''

if __name__ == "__main__":
    main()