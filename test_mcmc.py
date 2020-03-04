from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd

def main2():
    size = 5
    init_toric = Toric_code(size)

    init_toric.generate_random_error(0.10)
    #init_toric.qubit_matrix = apply_stabilizers_uniform(init_toric.qubit_matrix)
    init_toric.syndrom('next_state')

    print(init_toric.qubit_matrix)

    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')

    parallel_tempering(init_toric, 15, p=0.10, steps=10000, iters=10, conv_criteria='majority_based')

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
    N = 15  # number of chains in ladder, must be odd
    try:
        N % 2 == 0
    except:
        print('Number of chains was not odd.')
    ladder = []  # ladder to store all chains
    p_start = 0.1  # what should this really be???
    p_end = 0.75  # p at top chain as per high-threshold paper
    tops0 = 0
    SEQ = 2
    TOPS = 5
    eps = 0.1
    convergence_reached = 0
    nbr_errors_bottom_chain = []
    ec_frequency = []

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

        bottom_equivalence_classes = np.zeros(steps, dtype=int)

        for j in range(steps):
            # run mcmc for each chain [steps] times
            for i in range(N):
                for _ in range(iters):
                    ladder[i].update_chain()
            # now attempt flips from the top down
            for i in reversed(range(N - 1)):
                if i == (N - 2):
                    ladder[i + 1].flag = 1
                if ladder[0].flag == 1:
                    tops0 += 1
                    ladder[0].flag = 0
                r_flip(ladder[i], ladder[i + 1])

            #  Konvergenskriterium 1 i papper
            temp = np.count_nonzero(ladder[0].toric.qubit_matrix)
            nbr_errors_bottom_chain.append(temp)  # vill man räkna y som två fel?
            if tops0 >= TOPS:
                second_quarter = nbr_errors_bottom_chain[(len(nbr_errors_bottom_chain) // 4): (len(nbr_errors_bottom_chain) // 4) * 2]
                fourth_quarter = nbr_errors_bottom_chain[(len(nbr_errors_bottom_chain) // 4) * 3: (len(nbr_errors_bottom_chain) // 4) * 4]
                Average_second_quarter = sum(second_quarter) / (len(second_quarter))
                Average_fourth_quarter = sum(fourth_quarter) / (len(fourth_quarter))
                error = abs(Average_second_quarter - Average_fourth_quarter)
                if convergence_reached == 1:
                    ec_frequency.append(define_equivalence_class(ladder[0].toric.qubit_matrix))
                if error > eps:
                    tops0 = TOPS
                if tops0 == TOPS + SEQ:
                    if convergence_reached == 0:
                        print('Convergence achieved.')
                    convergence_reached = 1
            # record current equivalence class in bottom layer

            bottom_equivalence_classes[j] = define_equivalence_class(ladder[0].toric.qubit_matrix)

        # plot all chains
        for i in range(N):
            ladder[i].plot('Chain_' + str(i))

        # count number of occurrences of each equivalence class
        # equivalence_class_count[i] is the number of occurences of equivalence class number 'i'
        # if
        equivalence_class_count = np.bincount(bottom_equivalence_classes, minlength=15)

        print('Equivalence classes: \n', np.arange(16))
        print('Count:\n', equivalence_class_count)

        saveData(toric_init.qubit_matrix, equivalence_class_count, 'hej')

        steps = input('How many steps? Ans: ')
        iters = input('How many iterations for each step? Ans: ')

'''
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
   file_path=os.path.join(os.getcwd(),
                       "data",
                       'df.csv'))


    df = pd.read_pickle(file_path)
    df = df.append(pd.DataFrame([[init_qubit_matrix, distr, params]]))
    df.to_pickle(file_path) #  går att använda json https://stackoverflow.com/questions/48428100/save-pandas-dataframe-with-numpy-arrays-column
    
    # packa upp
    df = pd.read_pickle('df.csv')
    for index, loaded_qubit_matrix, loaded_distr, loaded_params in df.itertuples():
        print('Now printing df:')
        print('Index: ' + str(index))
        print(loaded_qubit_matrix)
        print(loaded_distr)
        print(loaded_params)
   '''

if __name__ == '__main__':
    main2()
