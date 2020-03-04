import numpy as np
import random as rand
import copy

from .toric_model import Toric_code
from .util import Action
from .mcmc import *


def reward(toric:Toric_code, suggested_corr_chain:Chain,p_rate):
    init_toric=suggested_corr_chain.toric
    Nc=None 
    p=p_rate
    SEQ=2
    TOPS=10
    eps=0.1
    steps=10000
    iters=10
    conv_criteria='distr_based' # 'error_based' #### 'distr_based'
    weight=1

  
    terminal = np.all((toric.current_state-suggested_corr_chain.toric.current_state)==0)

    if terminal:
        
        temp = parallel_tempering(init_toric,Nc, p, SEQ, TOPS, eps, steps, iters, conv_criteria)
        eq_class_bins=temp[1]
        print(eq_class_bins)
        sugg_class=define_equivalence_class(suggested_corr_chain.toric.qubit_matrix)
        rewar = (eq_class_bins[sugg_class] / (np.sum(eq_class_bins)))*weight
    else:
        rewar=-100
    return rewar   