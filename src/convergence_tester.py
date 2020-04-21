import numpy as np
import random as rand
import copy
import time
import matplotlib.pyplot as plt
import sys
import os

from .toric_model import *
from .util import Action
from .mcmc import *

from .toric_model import Toric_code


def Nc_tester(file_path, Nc_interval=[3,31]):
    size = 5
    p_error = 0.15
    SEQ = 8
    Nc = 9
    TOPS = 10
    tops_burn = 5
    steps = 1000
    eps = 0.008
    iters = 10
    conv_criteria='error_based'

    stats = pd.DataFrame(columns=['Nc', 'time', 'steps'])

    # Number of times every parameter configuration is tested
    pop = 10

    t_list = []
    for i in range(pop):
        t = Toric_code(5)
        t.generate_random_error(p_error)
        t_list.append(t)

    k=(Nc_interval[1] - Nc_interval[0]) / 2 + 1

    for Nc in range(Nc_interval[0], Nc_interval[1], 2):
        for pt in range(pop):
            t1 = time.time()
            _, conv_step = parallel_tempering_plus(copy.deepcopy(t_list[pt]), Nc=Nc, p=p_error, SEQ=SEQ, TOPS=TOPS, tops_burn=tops_burn, eps=eps, steps=steps, iters=iters, conv_criteria=conv_criteria)
            delta_t = time.time() - t1

            tmp_dict = {'Nc': Nc, 'time': delta_t, 'steps': conv_step}

            stats = stats.append(tmp_dict, ignore_index=True)

    stats.to_pickle(file_path)

    
# Runs test_numeric_distribution_convergence for an array of different tolerences
def compare_graphic(convergence_criteria='error_based',tolerences=[1.6,0.8,0.4,0.2,0.1,0.05],SEQs=[2,1,0]):
    x = tolerences
    y = np.zeros(len(tolerences))
    y2 = np.zeros(len(tolerences))
    for j in range(len(SEQs)):
        for i in range(len(tolerences)):
            [_,_,_,tmp,time_temp]=test_numeric_distribution_convergence(convergence_criteria,SEQs[j],x[i],x[i],False)
            y[i]=tmp
            y2[i]=time_temp*0.000001
        
        subplot_size=np.ceil(np.sqrt(2*len(SEQs)))
        plt.subplot(subplot_size,subplot_size,2*j+1)
        plt.title(convergence_criteria + "SEQ=" +str(SEQs[j])) 
        plt.xlabel("tolerence") 
        plt.ylabel("max diff") 
        plt.plot(x,y) 

        plt.subplot(subplot_size,subplot_size,2*j+2)
        plt.title(convergence_criteria + "SEQ=" +str(SEQs[j])) 
        plt.xlabel("tol.") 
        plt.ylabel("# steps (e+5)") 
        plt.plot(x,y2) 
    plt.show()

# Returns an array [nmbr_1st,nmbr_2nd,nmbr_3rd,max_1], and saves the same data in .txt files
# For a given convergence criteria and tolerence. nmbr_1st is the number of different seeds( maximum 16) which converges to
# a distribution that has the same most likely equivalence class (1st likely). Same way for nmbr_2nd and nmbr_3rd.
# max_1 is the maximum total variational distance between the different equivalence classes.
def test_numeric_distribution_convergence(convergence_criteria='distr_based',SEQ=2,eps=0.1,n_tol=0.05,bool=False):
    arr, time_array=test_distribution_convergence(convergence_criteria,SEQ,eps,n_tol,False)
    #arr=np.ndarray([[0.5,0.3,0.05,0.15,0,1.15,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0],[0.5,0.3,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    nmbr_1st=0
    nmbr_2nd=0
    nmbr_3rd=0
    max_1=0
    max_time=max(time_array)
    # temp can be viewed as 16 different counters
    temp=np.zeros(16)
    temp2=0
    count=0

    # The amount of different pairs of distributions to be compared is sum 1-->16 =136
    max_temp=np.zeros(136)
    for i in range(16):
        temp2=arr[i]
        # idx is index of the maximum element of the i:th distribution
        [idx]=np.where(temp2==np.ndarray.max(temp2))
        # increase counter number idx by one
        temp[idx]=temp[idx]+1

        # Check total variatonal distance for all different seed combinations.
        for j in range(i,16):
            a=np.zeros(16)
            for k in range(16):
                #
                a[k]=np.absolute(arr[j][k]-arr[i][k])
                #DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG 
                #if j==12 and i==10:     
                    #print("Element 1: " + str(arr[j][k]))
                    #print("Element 2: " + str(arr[i][k]))
                    #kon=arr[j][k]-arr[i][k]
                    #print("Element 1 - Element 2: "+ str(kon))
                    #print("Absolute of difference: " + str(np.absolute(kon)))
                    #print("Absolute of difference,a[k]: " + str(a[k]))  
            
            # max_temp[count] is the total variational distance of the count:th combination.
            # Count goes from 0 to 135        
            max_temp[count]=max(a)
            count+=1 
    # max_1 is the maximum tvd of all 136 combinations.
    max_1=max(max_temp)
    # nmbr_1st is the number of different seeds( maximum 16) which converges to
    # a distribution that has the same most likely equivalence class (1st most likely)
    nmbr_1st=max(temp)

    # Reset temp to only zeros
    temp=np.zeros(16)
    for i in range(16):
        temp2=arr[i]
        # idx is index of the maximum element of the i:th distribution
        [idx]=np.where(temp2==np.ndarray.max(temp2))
        # set maximum to 0
        temp2[idx]=0
        # idx is index of the 2nd largest element of the i:th distribution
        [idx]=np.where(temp2==np.ndarray.max(temp2))
        # increase counter number idx by one
        temp[idx]=temp[idx]+1
    # nmbr_2nd is the number of different seeds( maximum 16) which converges to
    # a distribution that has the same second most likely equivalence class (2nd most likely)
    nmbr_2nd=max(temp)

    # Reset temp to only zeros
    temp=np.zeros(16)
    for i in range(16):
        temp2=arr[i]
        # idx is index of the 2nd largest element of the i:th distribution
        [idx]=np.where(temp2==np.ndarray.max(temp2))
        # set 2nd largest element to 0
        temp2[idx]=0
        # idx is index of the 3rd largest element of the i:th distribution
        [idx]=np.where(temp2==np.ndarray.max(temp2))
        # increase counter number idx by one
        temp[idx]=temp[idx]+1

    # nmbr_3rd is the number of different seeds( maximum 16) which converges to
    # a distribution that has the same third most likely equivalence class (3rd most likely)
    nmbr_3rd=max(temp)

    
    if convergence_criteria=='error_based':
        # Create new file eps value in the file name
        f=open("data_" + convergence_criteria + '_eps_' + str(eps) + '_SEQ_' + str(SEQ) + ".txt","w")
        # Write the used convergence criteria
        f.write("Convergence criteria: " + convergence_criteria)
        # Write the used tolerence
        f.write('\n eps= ' + str(eps) + '\n SEQ=' + str(SEQ))
        # Write the time it takes
        f.write("\nMaximum number of steps for convergence: " +str(max_time) + "\n \n")
    elif convergence_criteria=='distr_based':
        f=open("data_" + convergence_criteria + '_ntol_' + str(n_tol) + '_SEQ_' + str(SEQ) + ".txt","w")
        f.write("Convergence criteria: " + convergence_criteria)
        f.write('\n ntol= ' + str(n_tol) + '\n SEQ= ' + str(SEQ))
        f.write("\nMaximum number of steps for convergence: " +str(max_time) + "\n \n")

    #Write the different critera parameters in the file adn close it
    f.write("Number of equivalent 1st: " + str(nmbr_1st) + "\nNumber of equivalent 2nd: "+ str(nmbr_2nd) + "\nNumber of equivalent 3rd: "+ str(nmbr_3rd))
    f.write("\n\nMax difference: " + str(max_1))
    f.close()
    #Also return the different critera parameters
    return [nmbr_1st,nmbr_2nd,nmbr_3rd,max_1,max_time]
    
# Returns list with 16 rows. 
# Each row contains the equivalence class distribution. 
# Each row uses a seed from a different equivalence class.
def test_distribution_convergence(convergence_criteria='distr_based',SEQ=2,eps=0.1,n_tol=0.05,boll=False):
    array_of_distributions=[]
    array_of_time=np.zeros(16)
    #Does paralell tempering for each of the 16 equivalence classes
    for i in range(16):
        # Choose seed
        t = seed(i+1)
        temp,count= parallel_tempering_plus(t, Nc=9, p=0.1, SEQ=SEQ, TOPS=10, tops_burn=2, eps=eps,n_tol=n_tol, steps=1000000, iters=10, conv_criteria=convergence_criteria)
        #convert to int32 because parallel_tempering returns only unsigned ints from 0 to 256
        temp=temp.astype(np.int32)
        #add the distribution from the current seed to the list to be returned
        array_of_distributions += [temp]
        array_of_time[i]=count
    
    #boolean parameter boll, if boll==True, then prints the distributions of the 16 equivalence classes as 16 subplots.
    if boll:
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

    return array_of_distributions, array_of_time
    

def convergence_analysis():
    size = 5
    p_error = 0.15
    Nc = 9
    TOPS=10
    tops_burn=5
    steps=500000

    # Number of times every parameter configuration is tested
    pop = 10

    #criteria = ['error_based', 'distr_based', 'majority_based', 'tvd_based', 'kld_based']
    criteria = ['error_based']

    SEQ_list = [i for i in range(2, 11, 2)]

    lst = np.array([1, 2, 4, 7, 10])
    eps_list = lst*1e-3
    n_tol_list = lst * 1e-3
    tvd_tol_list = lst * 8e-4 + 2e-2
    kld_tol_list = lst * 5e-3 + 5e-1
    
    lists = {'error_based': eps_list, 'distr_based': n_tol_list, 'majority_based': [0]*len(lst), 'tvd_based': tvd_tol_list, 'kld_based': kld_tol_list}

    tols = {crit: lists[crit] for crit in criteria}

    def tvd(a, b):
        return np.amax(np.absolute(a - b))

    def kld(a, b):
        nonzero = np.logical_and(a != 0, b != 0)
        if np.any(nonzero):
            log = np.log2(np.divide(a, b, where=nonzero), where=nonzero)
            return np.sum((a - b) * log, where=nonzero)
        else:
            return -1

    #crits_stats = {crit: [[], [], [], [], []] for crit in criteria}
    crits_stats = pd.DataFrame(columns=['criteria', 'SEQ', 'tol', 'kld', 'tvd', 'steps'])

    t1 = time.time()

    for SEQ in SEQ_list:
        for i in range(len(lst)):
            for j in range(pop):
                print('##################################')
                print('SEQ: ', SEQ, 'i: ', i, 'j: ', j)

                init_toric = Toric_code(size)
                init_toric.generate_random_error(p_error)

                args = {}
                if 'error_based' in criteria:
                    eps = tols['error_based'][i]
                    args['eps'] = eps
                
                if 'distr_based' in criteria:
                    n_tol = tols['distr_based'][i]
                    args['n_tol'] = n_tol

                if 'tvd_based' in criteria:
                    tvd_tol = tols['tvd_based'][i]
                    args['tvd_tol'] = tvd_tol

                if 'kld_based' in criteria:
                    kld_tol = tols['kld_based'][i]
                    args['kld_tol'] = kld_tol

                [distr, eq, eq_full, chain0, burn_in, crits_distr] = parallel_tempering_analysis(init_toric, Nc, p=p_error, TOPS=TOPS, SEQ=SEQ, tops_burn=tops_burn, steps=steps, conv_criteria=criteria, **args)

                distr = np.divide(distr.astype(np.float), 100)

                for crit in criteria:
                    tvd_crit = tvd(distr, crits_distr[crit][0])
                    kld_crit = kld(distr, crits_distr[crit][0])
                    print('==============================================')
                    print(crit)
                    print('convergence step: ', crits_distr[crit][1])
                    print('kld: ', kld_crit)
                    print('tvd:', tvd_crit)
                    tmp_dict = {'criteria': crit, 'SEQ': SEQ, 'tol': tols[crit][i], 'kld': kld_crit, 'tvd': tvd_crit, 'steps': crits_distr[crit][1]}
                    crits_stats = crits_stats.append(tmp_dict, ignore_index=True)

        print('SEQ: ', SEQ, ', Time completed: ', time.time() - t1)
    
    file_name = 'conv_test_SEQ_' + str(SEQ_list[0]) + '_' + str(SEQ_list[-1]) + '_p_' + str(p_error) + '.xz'
    crits_stats.to_pickle(file_name)


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

#Initial seed. Is used in seed(number) to generate seeds from all 16 eq-classes
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
'''       
convergence_criteria=str(sys.argv[1])
if convergence_criteria=='distr_based':
    eps=10
    n_tol=float(sys.argv[2])
elif convergence_criteria=='error_based':
    eps=float(sys.argv[2])
    n_tol=10

print("Convergence critera: " + convergence_criteria + ", eps: "+ str(eps) +", n_tol: " + str(n_tol))
[tmp,_,_,_]=test_numeric_distribution_convergence(convergence_criteria,eps,n_tol,bool=True)
print("Number of seed with same largest bin: " + str(tmp))   
'''

if __name__ == '__main__':
    try:
        array_id = str(sys.argv[1])
        local_dir = str(sys.argv[2])
        timeout = int(sys.argv[3])
    except:
        array_id = '0'
        local_dir = '.'
        timeout = 100000000000
        print('invalid sysargs')

    # Build file path
    file_path = os.path.join(local_dir, 'Nc_data_' + array_id + '.xz')

    Nc_tester(file_path=file_path, Nc_interval=[3, 31])
