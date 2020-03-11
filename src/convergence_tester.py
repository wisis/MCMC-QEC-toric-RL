import numpy as np
import random as rand
import copy
import time
import matplotlib.pyplot as plt

from .toric_model import *
from .util import Action
from .mcmc import *
from .reward import *


def test_tolerence_all_seeds(eps_interval=[0,0.1],norm_tol=[0.09,0.2],convergence_criteria='distr_based'):
    y1=[]
    y2=[]
    x=[]
    for i in range(1):
        for j in range(6):
            time_array=time_all_seeds(convergence_criteria,eps_interval[0],norm_tol[0]+j*(norm_tol[1]-norm_tol[0])/5)
            y1+=[time_array[1]]
            y2+=[time_array[2]]
            x+=[norm_tol[0]+j*(norm_tol[1]-norm_tol[0])/5]
    plt.plot(x, y1, label = "line 1")
    plt.plot(x, y2, label = "line 2")
    plt.show()
            
    
    
def test_distribution_convergence(convergence_criteria='distr_based',eps=0.1,n_tol=2):
    torr=seed(1)
    print('Equivalence class of seed(1): ' + str(define_equivalence_class(torr.qubit_matrix)))
    array_of_distributions=[]
    for i in range(16):
        t = seed(i+1)
        [_,_,temp,_] = parallel_tempering(t, 9, 0.1, 2, 10, eps,n_tol, 100000, 10, convergence_criteria)
        array_of_distributions += [temp]
    x = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] 
    for i in range(16):
        height = array_of_distributions[i]
        height=np.divide(height,np.sum(height))
        tick_label = ['1', '2', '3', '4', '5','6','7','8','9','10','11','12','13','14','15','16']
        plt.subplot(4,4,i+1) 
        plt.bar(x, height, tick_label = tick_label, width = 0.8, color = ['red', 'green'])
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.xlabel('Equivalence classes') 
        plt.ylabel('y - axis') 
        plt.title('') 
        
    plt.show() 
    return array_of_distributions
    
def time_all_seeds(convergence_criteria='distr_based',eps=0.1,n_tol=0.5):
    torr=seed(1)
    print('Equivalence class of seed(1): ' + str(define_equivalence_class(torr.qubit_matrix)))
    time_array=[]
    for i in range(16):
        t=seed(i+1)
        #tchain.toric.plot_toric_code(tchain.toric.current_state,"Hej")
        time_1=time.time()
        parallel_tempering(t, 9, 0.1, 2, 10, eps,n_tol, 1000, 10, convergence_criteria)
        time_2=time.time()
        time_array+=[(time_2-time_1)]
    x = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16] 
    height = time_array
    tick_label = ['1', '2', '3', '4', '5','6','7','8','9','10','11','12','13','14','15','16'] 
    plt.bar(x, height, tick_label = tick_label, 
        width = 0.8, color = ['red', 'green'])
    plt.xlabel('Equivalence classes') 
    plt.ylabel('y - axis') 
    plt.title('Time for converging from different equivalence classes') 
    plt.show() 
    return time_array

def in_seed(seed_number):
    toric=Toric_code(5)
    n=seed_number
    if n<1 or n>1:
        print('You tried to get a non-valid seed')
    elif n==1:
        action = Action(position = np.array([0, 2, 0]), action = 1) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
        toric.step(action)#1
        action = Action(position = np.array([0, 0, 2]), action = 2)
        toric.step(action)#2
        action = Action(position = np.array([0, 0, 3]), action = 2)
        toric.step(action)#3
        action = Action(position = np.array([1, 0, 4]), action = 3)
        toric.step(action)#4
        action = Action(position = np.array([0, 1, 4]), action = 1)
        toric.step(action)#5
        action = Action(position = np.array([1, 2, 3]), action = 2) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
        toric.step(action)#6
        action = Action(position = np.array([0, 2, 2]), action = 2)
        toric.step(action)#7
        action = Action(position = np.array([0, 2, 3]), action = 1)
        toric.step(action)#8
        action = Action(position = np.array([0, 4, 1]), action = 2)
        toric.step(action)#9
        return toric
def seed(number):
    toric=in_seed(1)
    n=number
    if n<1 or n>16:
        print('You tried to get a non-valid seed')
    elif n==1:
        return toric
    elif n==2:
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,1)
        return toric
    elif n==3:
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,3)
        return toric
    elif n==4:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,1)
        return toric
    elif n==5:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,3)
        return toric
    elif n==6:
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,1)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,3)
        return toric
    elif n==7:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,1)
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,3)
        return toric
    elif n==8:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,1)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,1)
        return toric
    elif n==9:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,3)
        return toric
    elif n==10:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,1)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,3)
        return toric
    elif n==11:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,1)
        return toric
    elif n==12:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,1)
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,1)
        return toric
    elif n==13:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,1)
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,3)
        return toric
    elif n==14:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,1)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,1)
        return toric
    elif n==15:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,1)
        return toric
    elif n==16:
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,1)
        [toric.qubit_matrix,_]=apply_logical_vertical(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,3)
        [toric.qubit_matrix,_]=apply_logical_horizontal(toric.qubit_matrix,1,1)
        return toric
    
    