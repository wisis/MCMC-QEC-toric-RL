from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd

def main():
    size = 5
    toric_init = Toric_code(size)

    '''
    # Initial error configuration
    init_errors = np.array([[1, 0, 1, 3], [1, 1, 0, 1], [1, 3, 3, 3], [1, 4, 1, 1], [1, 1, 1, 1]])
    for arr in init_errors:  # apply the initial error configuration
        print(arr)
        action = Action(position=arr[:3], action=arr[3])
        toric_init.step(action)
    '''
    # create the diffrent chains in an array
    N = 15  # number of chains in ladder
    ladder = []  # ladder to store all chains
    p_start = 0.1  # what should this really be???
    p_end = 0.75  # p at top chain as per high-threshold paper

    # test random error initialisation
    toric_init.generate_random_error(p_start)
    toric_init.qubit_matrix = apply_stabilizers_uniform(toric_init.qubit_matrix)

    # plot initial error configuration
    toric_init.plot_toric_code(toric_init.next_state, 'Chain_init')

    # add and copy state for all chains in ladder
    for i in range(N):
        p_i = p_start + ((p_end - p_start) / (N - 1)) * i
        ladder.append(Chain(size, p_i))
        ladder[i].toric = copy.deepcopy(toric_init)  # give all the same initial state
    ladder[N - 1].p_logical = 0.5  # set top chain as the only one where logicals happen

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

        error_samples = np.zeros(steps, dtype=int)

        for j in range(steps):
            # run mcmc for each chain [steps] times
            for i in range(N):
                for _ in range(iters):
                    ladder[i].update_chain()
            # now attempt flips from the top down
            for i in reversed(range(N - 1)):
                r_flip(ladder[i], ladder[i + 1])
            # record current equivalence class in bottom layer

            error_samples[j] = define_equivalence_class(ladder[0].toric.qubit_matrix)

        # plot all chains
        for i in range(N):
            ladder[i].plot('Chain_' + str(i))

        # count number of occurrences of each equivalence class
        # equivalence_class_count[i] is the number of occurences of equivalence class number 'i'
        # if
        equivalence_class_count = np.bincount(error_samples, minlength=15)

        print('Equivalence classes: \n', np.arange(16))
        print('Count: \n', equivalence_class_count)

        saveData(toric_init.qubit_matrix, equivalence_class_count, 'hej')

        steps = input('How many steps? Ans: ')
        iters = input('How many iterations for each step? Ans: ')


def saveData(init_qubit_matrix, distr, params):
    # Sparar data från XXX antal mcmc körningar (typ 10000 steps/till konvergens med 10 iters)
    # En entry här motsvarar en träningsentry och innehåller därför följande:
    #  * Initial felkedja, denna krävs för att vi ska kunna köra DRL algoritmen på en faktisk felkedja.
    #  //Syndrom behövs INTE
    #  * Fördelning över ekvivalensklasser, typ 16array med fördelningssiffror, behövs som input till reward() för att
    #           se om vår lösningskedja tillhör rätt ekvivalensklass
    #  * Använda parametrar för generering av datapunkt

    #df = pd.DataFrame({ 'qubit_matrix': init_qubit_matrix,
    ##                    'distr': distr,
     #                   'params': params})

    df = pd.read_pickle('df.csv')
    df = df.append(pd.DataFrame([[init_qubit_matrix, distr, params]]))
    df.to_pickle('df.csv') # går att använda json https://stackoverflow.com/questions/48428100/save-pandas-dataframe-with-numpy-arrays-column
    
    # packa upp
    df = pd.read_pickle('df.csv')
    for index, loaded_qubit_matrix, loaded_distr, loaded_params in df.itertuples():
        print('Now printing df:')
        print('Index: ' + str(index))
        print(loaded_qubit_matrix)
        print(loaded_distr)
        print(loaded_params)
   

if __name__ == '__main__':
    main()
