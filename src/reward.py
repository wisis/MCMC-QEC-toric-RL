import numpy as np
import random as rand
import copy

from .toric_model import Toric_code
from .util import Action
from .mcmc import *


def reward(current_state, suggested_corr_chain, burn, p):

    eq_count = np.zeros(16)

    size = suggested_corr_chain.toric.system_size
    toric_init = copy.deepcopy(suggested_corr_chain.toric)

    # create the diffrent chains in an array
    N = current_state.system_size * 2 + 1  # number of chains in ladder
    ladder = []  # ladder to store all chains
    p_start = p  # what should this really be???
    p_end = 0.75  # p at top chain as per high-threshold paper

    # add and copy state for all chains in ladder
    for i in range(N):
        ladder.append(Chain(size, p_start + ((p_end - p_start) / (N - 1)) * i))
        ladder[i].toric = copy.deepcopy(toric_init)  # give all the same initial state
    ladder[N - 1].p_logical = 0.5  # set top chain as the only on where logicals happen

    steps = input('How many steps? Ans: ')
    iters = input('How many iterations for each step? Ans: ')

    try:
        steps = int(steps) + burn
        iters = int(iters)
    except:
        print('Input data bad, using default of 1 for both.')
        steps = 1
        iters = 1

    for k in range(steps):
        # run mcmc for each chain [iters] times
        for i in range(N):
            for _ in range(iters):
                ladder[i].update_chain()

        # now attempt flips from the top down
        for j in reversed(range(N - 1)):
            r_flip(ladder[j], ladder[j + 1])

        if k >= burn:
            eq_last = define_equivalence_class(ladder[0].toric.qubit_matrix)
            eq_count[eq_last] += 1

    ec_sum = np.sum(eq_count)
    sugg_class = define_equivalence_class(suggested_corr_chain.toric.qubit_matrix)

    #  Uncoment if you wanna print the distribution between eq. classes
    #  for i in range(16):
        #  print("Class: " + str(i))
        #  print(eq_count[i]/ec_sum)

    return eq_count[sugg_class] / ec_sum