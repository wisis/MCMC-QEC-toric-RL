from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy

def main():
    size = 3
    toric = Toric_code(size)

    init_errors = np.array([[1, 0, 2, 1], [1, 0, 1, 1], [1, 0, 3, 1], [[1, 0, 4, 1]], [[1, 0, 0, 1]]])
    for arr in init_errors:
        print(arr)
        action = Action(position = arr[:3], action = arr[3])
        toric.step(action)

    toric.plot_toric_code(toric.next_state, 'Chain_0')

    apply_n_independent_random_stabilizers(toric, 1000)

    qubit_matrix = np.copy(toric.qubit_matrix)
    p = 0.1
    p_logical = 0.5

    #for i in range(100):
    #    qubit_matrix = permute_error(qubit_matrix, size, p)

    res = input('plot? ')
    while not res == 'n':
        try:
            n = int(res)
        except:
            n = 1
        for i in range(n):
            qubit_matrix = permute_error(qubit_matrix, size, p, p_logical)
            toric.qubit_matrix = np.copy(qubit_matrix)
        toric.plot_toric_code(toric.next_state, 'Chain_0')
        res = input('plot? ')
    #print("vertex: " + str(toric.next_state[0, :, :]))
    #print("plaquette: " + str(toric.next_state[1, :, :])) 


def main2():
    size = 5
    toric_init = Toric_code(size)

    # Initial error configuration
    init_errors = np.array([[1, 0, 1, 1], [1, 2, 1, 1], [1, 3, 1, 1], [1, 4, 1, 1], [1, 1, 1, 1]])
    for arr in init_errors:  # apply the initial error configuration
        print(arr)
        action = Action(position=arr[:3], action=arr[3])
        toric_init.step(action)
    # plot initial error configuration
    toric_init.plot_toric_code(toric_init.next_state, 'Chain_init')

    # create the diffrent chains in an array
    N = 9  # number of chains in ladder
    ladder = []  # ladder to store all chains
    p_start = 0.1  # what should this really be???
    p_end = 0.75  # p at top chain as per high-threshold paper

    # add and copy state for all chains in ladder
    for i in range(N):
        ladder.append(Chain(size, p_start + ((p_end - p_start) / (N - 1)) * i))
        ladder[i].toric = copy.deepcopy(toric_init)  # give all the same initial state
    ladder[N - 1].set_p_logical(0.5)  # set top chain as the only on where logicals happen

    steps = input('How many steps? Ans: ')
    iters = input('How many iterations for each step? Ans: ')
    while not (steps == 'n' or iters == 'n'):
        try:
            steps = int(steps)
            iters = int(iters)
        except:
            print('Input data bad, using default of 1 for both.')
            steps = 1
            iters = 1

        for _ in range(iters):
            # run mcmc for each chain [steps] times
            for i in range(N):
                for _ in range(steps):
                    ladder[i].permute_error()
            # now attempt flips from the top down
            for j in reversed(range(N - 1)):
                r_flip(ladder[j], ladder[j + 1])

        # plot all chains
        for i in range(N):
            ladder[i].plot('Chain_' + str(i))

        steps = input('How many steps? Ans: ')
        iters = input('How many iterations for each step? Ans: ')


'''
    rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)
    opps = np.array([0, 1, 2])
    rules = rule_table[1][opps]
    print(rules)
'''

if __name__ == "__main__":
    main2()
