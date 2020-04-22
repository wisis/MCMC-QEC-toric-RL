from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd
import time
import matplotlib.pyplot as plt


def convergence_tester():
    size = 5
    init_toric = Toric_code(size)
    Nc = 9
    p_error = 0.17
    success = 0
    correspondence = 0
    
    for i in range(1000):
          t1 = time.time()
          init_toric.generate_random_error(p_error)
          toric_copy = copy.deepcopy(init_toric)
          apply_random_logical(toric_copy.qubit_matrix)
          class_before = define_equivalence_class(init_toric.qubit_matrix)
          [distr1, _, _, _, _] = parallel_tempering(init_toric, 9, p=p_error, steps=1000000, iters=10, conv_criteria='error_based')
          [distr2, _, _, _, _] = parallel_tempering(toric_copy, 9, p=p_error, steps=1000000, iters=10, conv_criteria='error_based')
          class_after = np.argmax(distr1)
          copy_class_after = np.argmax(distr2)
          if class_after == class_before:
              success+=1
          if copy_class_after == class_after:
              correspondence+=1
          
          if i >= 1:
              print('#' + str(i) + " current success rate: ", success/(i+1))
              print('#' + str(i) + " current correspondence: ", correspondence/(i+1), " time: ", time.time()- t1)


def main3(): # P_s som funktion av p
    points = 20
    size = 5
    init_toric = Toric_code(size)
    Nc = 19
    TOPS=20
    SEQ=30
    tops_burn=10
    eps=0.008
    steps=1000000
    p_error = [i*0.01 + 0.05 for i in range(points)]

    # define error
    '''
    action = Action(position = np.array([1, 1, 0]), action = 2) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([1, 2, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 3, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 4, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 6, 2]), action = 3)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 6, 3]), action = 3)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 6, 4]), action = 3)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 6, 1]), action = 2)
    init_toric.step(action)#2
    '''

    init_toric.qubit_matrix = np.array([[[0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [1, 0, 0, 0, 1],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0]],
                                        [[0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0]]])

    #init_toric.generate_n_random_errors(9)

    init_toric.syndrom('next_state')
    


    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    t1 = time.time()

    startingqubit = init_toric.qubit_matrix

    data = []

    for i in range(points):
        #init_toric.qubit_matrix, _ = apply_random_logical(startingqubit)

        distr = parallel_tempering(init_toric, Nc=Nc, p=p_error[i], steps=steps, SEQ=SEQ, TOPS=TOPS, tops_burn=tops_burn, eps=eps, conv_criteria='error_based')
        #print("error #" + str(i) + ': ', eq_class_count_BC/np.sum(eq_class_count_BC))
        distr_i = np.divide(distr, np.sum(distr), dtype=np.float)
        data.append(distr_i)
        print(p_error[i], distr_i)
    
    data = np.asarray(data)
    print(data[:,0])
    for i in range(16):
        plt.plot(p_error, data[:,i], label=('eq_class_' + str(i+1)))
    plt.xlabel('Error rate, p')
    plt.ylabel('Probability of equivalance class')
    plt.title('init: k3')
    plt.legend(loc=1)

    plt.show()
        
    print("runtime: ", time.time()-t1)


def eq_evolution():
    size = 5
    init_toric = Toric_code(size)
    p_error = 0.1
    Nc = 15
    steps=10000
    
    # define error
    action = Action(position = np.array([1, 1, 0]), action = 2) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([1, 2, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 3, 0]), action = 1)
    init_toric.step(action)#2

    # eller använd någon av dessa för att initiera slumpartat
    #nbr_error = 9
    #init_toric.generate_n_random_errors(nbr_error)
    #init_toric.generate_random_error(0.10)
    init_toric.syndrom('next_state')


    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    t1 = time.time()

    starting_qubit = init_toric.qubit_matrix

    for i in range(2):
        init_toric.qubit_matrix, _ = apply_random_logical(starting_qubit)

        [distr, eq, eq_full, chain0, resulting_burn_in] = parallel_tempering(init_toric, Nc, p=p_error, steps=steps, iters=10, conv_criteria=None)

        mean_history = np.array([eq[x] / (x + 1) for x in range(steps)])

        plt.plot(mean_history)
        plt.savefig('plots/history_'+str(i+1)+'.png')

    print("runtime: ", time.time()-t1)

def convergence_analysis():
    size = 5
    init_toric = Toric_code(size)
    p_error = 0.185
    Nc = 19
    TOPS=20
    SEQ=30
    tops_burn=10
    eps=0.008
    n_tol=1e-4
    steps=1000000

    criteria = ['error_based'] #, 'distr_based', 'majority_based']

    # define error
    #init_toric.qubit_matrix[1, 1, 0] = 2
    #init_toric.qubit_matrix[1, 2, 0] = 1
    #init_toric.qubit_matrix[1, 3, 0] = 1

    init_toric.qubit_matrix = np.array([[[0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 1, 2, 1, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0]],
                                        [[0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 0, 2, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0]]])

    # eller använd någon av dessa för att initiera slumpartat
    #nbr_error = 9
    #init_toric.generate_n_random_errors(nbr_error)
    #init_toric.generate_random_error(0.10)
    init_toric.syndrom('next_state')

    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init', define_equivalence_class(init_toric.qubit_matrix))
    t1 = time.time()

    init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)

    [distr, eq, eq_full, chain0, burn_in, crits_distr] = parallel_tempering_analysis(init_toric, Nc, p=p_error, TOPS=TOPS, SEQ=SEQ, tops_burn=tops_burn, eps=eps, n_tol=n_tol, steps=steps, conv_criteria=criteria)

    mean_history = np.array([eq[x] / (x + 1) for x in range(steps)])

    for i in range(16):
        plt.plot(mean_history[: , i], label=i)
    print('Steps to burn in: ', burn_in)
    for crit in criteria:
        print('==============================================')
        print(crit)
        print('convergence step: ', crits_distr[crit][1])
        print('converged distribution: ', crits_distr[crit][0])
        #plt.axvline(x=crits_distr[crit][1], label=crit)

    plt.legend(loc=1)
    plt.show()


if __name__ == '__main__':
    #convergence_tester()
    #eq_evolution()
    #convergence_analysis()
    main3()