from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd
import time
import matplotlib.pyplot as plt

def plot_result():
    size = 5
    Nc = 15
    max_steps=1000000
    points = 10
    trials = 1000

    succeses = np.zeros(points)
    p = np.linspace(0.05, 0.20, points)
    for i in range(points):
        print('now doing p: ', p[i])
        for _ in range(trials):
            init_toric = Toric_code(size)
            init_toric.generate_random_error(p[i])
            [distr, chains] = parallel_tempering(init_toric, Nc, p=p[i], steps=max_steps, iters=10, conv_criteria='error_based')
            if define_equivalence_class(init_toric.qubit_matrix) == define_equivalence_class(chains[np.argmax(distr)]):
                succeses[i] += 1
        print(succeses)
    plt.plot(p,succeses)
    plt.xlabel('p, error rate')
    plt.ylabel('P_s, success rate of algorithm')
    plt.title('0.05-0.20, 16 pts, 1 trials, default error based, d = 5, Nc = 15')
    plt.show()


if __name__ == '__main__':
    plot_result()