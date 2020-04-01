import numpy as np
import random as rand
import copy
import time
import matplotlib.pyplot as plt
import sys

from .toric_model import *
from .util import Action
from .mcmc import *
from .reward import *

from .toric_model import Toric_code
from .mcmc import *





'''
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
'''
'''            
def sucess_and_correspondence_tester(seed_number,iterations=1000):
    
    init_toric =seed(seed_number)
    size = init_toric.system_size
    Nc = 9
    p_error = 0.17
    success = 0
    correspondence = 0
    
    for i in range(iterations):
          
          toric_copy = copy.deepcopy(init_toric)
          apply_random_logical(toric_copy.qubit_matrix)
          class_before = define_equivalence_class(init_toric.qubit_matrix)
          [distr1, eq_class_count_BC,eq_class_count_AC,chain0] = parallel_tempering(init_toric, 9, p=p_error, steps=1000000, iters=10, conv_criteria='error_based')
          [distr2, eq_class_count_BC,eq_class_count_AC,chain0] = parallel_tempering(toric_copy, 9, p=p_error, steps=1000000, iters=10, conv_criteria='error_based')
          class_after = np.argmax(distr1)
          copy_class_after = np.argmax(distr2)
          if class_after == class_before:
              success+=1
          if copy_class_after == class_after:
              correspondence+=1
          
          if i >= 1:
              print('#' + str(i) +" current success rate: ", success/(i+1))
              print('#' + str(i) + " current correspondence: ", correspondence/(i+1))   
  ''' 
def test_numeric_distribution_convergence(convergence_criteria='distr_based',eps=0.1,n_tol=0.05,bool=False):
    arr=test_distribution_convergence(convergence_criteria,eps,n_tol,True)
    nmbr_1st=0
    nmbr_2nd=0
    nmbr_3rd=0
    temp=np.zeros(16)
    temp2=0
    for i in range(16):
        temp2=arr[i]
        idx=temp2.index(max(temp2))
        temp[idx]=temp[idx]+1
    nmbr_1st=max(temp)
    temp=np.zeros(16)
    for i in range(16):
        temp2=arr[i]
        idx=temp2.index(max(temp2))
        temp2[idx]=0
        idx=temp2.index(max(temp2))
        temp[idx]=temp[idx]+1
    nmbr_2nd=max(temp)

    temp=np.zeros(16)
    for i in range(16):
        temp2=arr[i]
        idx=temp2.index(max(temp2))
        temp2[idx]=0
        idx=temp2.index(max(temp2))
        temp2[idx]=0
        idx=temp2.index(max(temp2))
        temp[idx]=temp[idx]+1
    nmbr_3rd=max(temp)
    f=open("data_" + convergence_criteria + '_eps' + str(eps)+ '_ntol' + str(n_tol))
    f.write("Number of equivalent 1st: " + str(nmbr_1st) + "\n Number of equivalent 2nd: "+ str(nmbr_2nd) + "\n Number of equivalent 3rd: "+ str(nmbr_3rd))
    
    

def test_distribution_convergence(convergence_criteria='distr_based',eps=0.1,n_tol=0.05,bool=False):
    torr=seed(1)
    print('Equivalence class of seed(1): ' + str(define_equivalence_class(torr.qubit_matrix)))
    array_of_distributions=[]
    for i in range(16):
        t = seed(i+1)
        print(i)
        [_,_,temp,_] = parallel_tempering(t, 9, 0.1, 2, 10, eps,n_tol, 100000, 10, convergence_criteria)
        array_of_distributions += [temp]
    x = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] 
    if bool:
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
    if n<1 or n>2:
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
    elif n==2: #Mats seed
        action = Action(position = np.array([1, 2, 1]), action = 1) 
        toric.step(action)#1
        action = Action(position = np.array([1, 3, 1]), action = 1) 
        toric.step(action)#2
        return toric
def seed(number):
    toric=in_seed(2)
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
        
convergence_criteria=str(sys.argv[1])
if convergence_criteria=='distr_based':
    eps=10
    n_tol=float(sys.argv[2])
else:
    eps=float(sys.argv[2])
    n_tol=10

print("eps: "+ str(eps) +", n_tol: " + str(n_tol))
print("Number of seed with same largest bin: " + str(test_numeric_distribution_convergence(convergence_criteria,eps,n_tol,bool=True)))   