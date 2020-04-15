from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

points = 5
p = np.linspace(0.14, 0.20, points)

def throw_away_function(i):
    size = 5
    Nc = 15
    max_steps=1000000

    init_toric = Toric_code(size)
    init_toric.generate_random_error(p[i])
    return parallel_tempering(init_toric, Nc, p=p[i], steps=max_steps, iters=10, conv_criteria='error_based')

def plot_result():
    trials = 100

    succeses = np.zeros(points)
    
    for i in tqdm(range(points)):
        print('now doing p: ', p[i])
        pool = mp.Pool(mp.cpu_count())

        for out in tqdm(pool.imap_unordered(throw_away_function, [i]*trials), total=trials, smoothing=0):
            succeses[i] += out
        print(succeses[i], '/', trials)
        print('P_s = ', succeses[i]/trials)

    plt.plot(p,succeses)
    plt.xlabel('p, error rate')
    plt.ylabel('P_s, success rate of algorithm')
    plt.title('0.14-0.20, 10 pts, 100 trials, default error based, d = 5, Nc = 15')
    plt.show()


if __name__ == '__main__':
    plot_result()