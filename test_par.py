import multiprocessing as mp
import numpy as np
from src.mcmc import *

'''
class Test_class:
    def __init__(self, n):
        self.array = n

def test(test_class, pipes):
    upper_pipe = pipes[0]
    lower_pipe = pipes[1]
    
    old_array = test_class.array
    if upper_pipe:
        if upper_pipe.poll(5):
            #higher_n_array = upper_pipe.recv()
            test_class.array = upper_pipe.recv()

        else:
            print('fuck', test_class.array)
            return
        upper_pipe.send(old_array)
    else: 
        higher_n_array = 1
    
    if lower_pipe:
        lower_pipe.send(test_class.array)
        if lower_pipe.poll(5):
            #lower_n_array = lower_pipe.recv()
            test_class.array = lower_pipe.recv()
        else:
            print('fuck', test_class.array)
            return
    else:
        lower_n_array = 1

    #test_class.array = test_class.array*higher_n_array*lower_n_array
    return test_class
'''

def parallel_tempering(ladder, steps, iters, pool):
    '''
    p_end = 0.75
    size = toric_init.system_size
    ladder = []
    for i in range(Nc):
        p_i = p + ((p_end - p) / (Nc - 1)) * i
        ladder.append(Chain(size, p_i))
        ladder[i].toric = copy.deepcopy(toric_init)  # give all the same initial state
    ladder[N - 1].p_logical = 0.5  # set top chain as the only one where logicals happen
    '''
    Nc = len(ladder)
    equivalence_class_samples = np.zeros(steps, dtype=int)

    '''
    #pool = mp.Pool(mp.cpu_count())
    pipe_list = [[None]*2 for i in range(Nc)]
    for i in range(Nc-1):
        pipes = mp.Pipe(True)
        pipe_list[i][0] = pipes[0]
        pipe_list[i+1][1] = pipes[1]

    result = pool.starmap(metropolis, [(ladder[i], pipe_list[i], steps, iters) for i in reversed(range(Nc))])
    '''
    for j in range(steps):
        ladder = pool.starmap(metropolis, [(ladder[i], iters) for i in range(Nc)])

        for i in reversed(range(Nc - 1)):
            r_flip(ladder[i], ladder[i + 1])   

        equivalence_class_samples[j] = define_equivalence_class(ladder[0].toric.qubit_matrix)
    #pool.close()
    return equivalence_class_samples


def metropolis(chain, iters):
    
    # Do a metropolis iteration
    for _ in range(iters):
        chain.update_chain()
    
    return chain


'''
def metropolis(chain, pipes, steps, iters):
    equivalence_class_samples = np.zeros(steps, dtype=int)

    upper_pipe = pipes[0]
    lower_pipe = pipes[1]

    #upper_chain = Chain(chain.size, 0)
    #lower_chain = Chain(chain.size, 0)
    
    for j in range(steps):
        # Do a metropolis iteration
        for _ in range(iters):
            chain.update_chain()
        
        # Uppermost chain doesn't have an upper_pipe
        if upper_pipe:
            # All other chains stop here to wait (10 s) for data from their upper chains
            if upper_pipe.poll(5):
                # Receive data from chain above
                upper_chain = upper_pipe.recv()
                # Attempt swap. If swap successful, upper_pipe and chain switch qubit_matrix
                r_flip(chain, upper_chain)
                # Send upper_chain back
                # Maybe only do this if a swap was made?----------------------------------------------
                upper_pipe.send(upper_chain)
            else:
                # If poll times out something is wrong, end function
                return
        
        # Lowest chain doesn't have a lower_pipe.
        # Uppermost chain starts here, sends its data down
        if lower_pipe:
            # Send data to chain below
            lower_pipe.send(chain)
            # Waits (10 s) for data from lower chain. Data is sent when a swap has been attempted
            if lower_pipe.poll(5):
                # Update chain when data received
                # Maybe only do this part if swap was actually made -----------------------------------
                chain = lower_pipe.recv()
            else:
                return

        if not lower_pipe:
            equivalence_class_samples[j] = define_equivalence_class(chain.toric.qubit_matrix)

    if not lower_pipe:
        return equivalence_class_samples
    return
'''
'''
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    n = 5
    lst = [Test_class(n+1) for n in reversed(range(n))]
    #lst = range(1, n+1)
    pipe_list = [[None]*2 for i in range(n)]
    for i in range(n-1):
        pipes = mp.Pipe(True)
        pipe_list[i][1] = pipes[1]
        pipe_list[i+1][0] = pipes[0]
    lst = pool.starmap(test, [(lst[i], pipe_list[i]) for i in range(n)])
    pool.close()
    print('\nLista:')
    for item in lst:
        print(item.array)
    for pipes in pipe_list:
        if pipes[0]:
            pipes[0].close()
'''