from src.mcmc import *
from src.toric_model import Toric_code
import numpy as np
import time

def main():
    size = 5
    p = 0.1
    bot = Chain(size, p)
    bot.toric.generate_random_error(p)
    bot.calc_num_errors()
    bot.plot('tst')
    for i in range(5):
        for j in range(10):
            bot.update_chain()
        bot.plot('tst' + str(i))

def main2():
    test_chain = Chain(5, 0.1)
    #test_chain.toric.generate_n_random_errors(25)
    test_chain.plot('test')
    new_matrix, error_change = apply_random_logical(test_chain.toric.qubit_matrix)
    copy_chain = copy.deepcopy(test_chain)
    copy_chain.toric.qubit_matrix = new_matrix
    copy_chain.plot('test_copy')
    print(error_change)
    

    '''
    arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print(arr)
    transp = arr.transpose(0, 2, 1)
    print(transp)
    transp[0, 1, 0] = 100
    print(arr)
    '''

def outer(x):
    def inner():
        z = 2 * x
        return z
    print(inner())

if __name__ == '__main__':
    main2()
