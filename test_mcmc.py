from src.toric_model import Toric_code
from src.mcmc import *
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
    
    init_errors = np.array([[1, 0, 1, 1], [1, 2, 1, 1], [1, 3, 1, 1], [1, 4, 1, 1], [1, 1, 1, 1]])
    for arr in init_errors:
        print(arr)
        action = Action(position = arr[:3], action = arr[3])
        toric_init.step(action)
    toric_init.plot_toric_code(toric_init.next_state,'Chain_init')
    #chain0.toric.plot_toric_code(toric.next_state, 'Chain_0')

    # Create N^2 chains (Toric_codes)
    # we are probably better off making the chains into objects in the mcmc class?
    # we can then create N^2 of them and initialize them with different params (like p)
    # and then easily perform swaps from here? or maybe that needs to move as well?


    # apply the random stabilizers (not really needed evenutally)
    # apply_n_independent_random_stabilizers(toric_init, 1000)


    # create the diffrent chains in an array
    N = 9
    ladder = []
    p_start = 0.1  # what should this really be???
    p_end = 0.75

    for i in range(N):
        ladder.append(Chain(size, p_start + ((p_end - p_start) / (N - 1)) * i))
        ladder[i].toric = copy.deepcopy(toric_init)  # give all the same initial state
    ladder[N-1].set_p_logical(0.75)  # set top chain as the only on where logicals happen

    res = input('Number of iterations? "n" to quit. Ans: ')
    while not res == 'n':
        for _ in range(1000):
            try:
                n = int(res)
            except:
                n = 1
            # run mcmc
            for i in range(N):
                for _ in range(n):
                    ladder[i].permute_error()
            # now attempt flips
            print('-------')
            for j in reversed(range(N-1)):
                result = r_flip(ladder[j],ladder[j+1])
            # plot
            
        for i in range(N):
                ladder[i].plot('Chain_' + str(i))
        res = input('Number of iterations? "n" to quit. Ans: ')
    

'''
    rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)
    opps = np.array([0, 1, 2])
    rules = rule_table[1][opps]
    print(rules)
'''

if __name__ == "__main__":
    main2()