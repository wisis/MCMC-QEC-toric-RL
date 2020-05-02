import numpy as np
import random as rand
import copy
import time
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

from .toric_model import *
from .util import Action
from .mcmc import *

from .toric_model import Toric_code


def geom_mean(series):
    array=series.to_numpy()
    return np.exp(np.average(np.log(array)))


def geom_std(series):
    array=series.to_numpy()
    return np.exp(np.std(np.log(array)))


def tvd(a, b):
    nonzero = np.logical_and(a != 0, b != 0)
    if np.any(nonzero):
        return np.amax(np.absolute(a - b))
    else:
        return -1


def kld(a, b):
    nonzero = np.logical_and(a != 0, b != 0)
    if np.any(nonzero):
        log = np.log2(np.divide(a, b, where=nonzero), where=nonzero)
        return np.sum((a - b) * log, where=nonzero)
    else:
        return -1


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
        print('Nc =', Nc, '/ ', Nc_interval[1])
        for pt in range(pop):
            t1 = time.time()
            _, conv_step = parallel_tempering_plus(copy.deepcopy(t_list[pt]), Nc=Nc, p=p_error, SEQ=SEQ, TOPS=TOPS, tops_burn=tops_burn, eps=eps, steps=steps, iters=iters, conv_criteria=conv_criteria)
            delta_t = time.time() - t1

            tmp_dict = {'Nc': Nc, 'time': delta_t, 'steps': conv_step}

            stats = stats.append(tmp_dict, ignore_index=True)

    stats.to_pickle(file_path)


def Nc_visuals(files=6):
    file_base = 'output/Nc_data_'

    stats = pd.DataFrame(columns=['Nc', 'time', 'steps'])
    for i in range(files):
        df = pd.read_pickle(file_base + str(i) + '.xz')
        stats = pd.concat([stats, df])

    agg_stats = ['Nc', 'time_mean', 'time_std', 'time_geom_mean', 'time_geom_std', 'steps_mean', 'steps_std', 'steps_geom_mean', 'steps_geom_std']

    agg_data = pd.DataFrame(columns = agg_stats)

    Nc_values = np.unique(stats['Nc'].to_numpy())

    # Number of data points
    tot_pts = stats.shape[0]
    # Number of different Nc values
    Nc_pts = Nc_values.size
    # Number of samples per (SEQ, tol) pair
    pop = int(tot_pts / Nc_pts)

    for Nc in Nc_values:
        # Window of points for current Nc value
        window = stats[stats['Nc'] == Nc]
        # Only look at converged runs
        window = window[window['steps'] != -1]
        # Calculate step time
        window['steptime'] = window['time'] / window['steps']
        # Mean and std values over converged runs
        agg = window.agg(['mean', 'std', geom_mean, geom_std])
        # number of converged runs
        nbr_converged = window.shape[0]
        # temporary dict to append to aggregated data
        tmp_dict = {'Nc': Nc, 'nbr_converged': nbr_converged}
        for name in ['time', 'steps', 'steptime']:
            for op in ['mean', 'std', 'geom_mean', 'geom_std']:
                tmp_dict[name + '_' + op] = agg.loc[op][name]

        # append aggregate data for current Nc value to agg_data
        agg_data = agg_data.append(tmp_dict, ignore_index=True)
    
    
    fig, host = plt.subplots(2)

    ax = host[0]
    converged = window['steps'] != -1

    nbr_converged = agg_data['nbr_converged']
    #yerr = [klds * (1 - 1/std_tvds), klds * (std_tvds - 1)]
    #ax.errorbar(tols, klds, yerr=yerr, label='Distance')
    ax.errorbar(Nc_values, agg_data['time_mean'].to_numpy(), yerr=agg_data['time_std'], label='Convergence time')
    #ax.scatter(Nc_values, agg_data['time_mean'], label='Convergence_time')

    #ax.set_title('Convergence steps and time as a function of Nc')
    ax.set_xlabel('Nc')
    #ax.set_xscale('log')
    ax.set_ylabel('Time [s]')
    ax.legend(loc='upper left')
    ax.set_ylim(0, agg_data['time_mean'].max()*1.5)
    #ax.set_yscale('log')
    
    par = ax.twinx()
    par.bar(Nc_values, agg_data['steps_mean'].to_numpy(), width=1.6, color='gray', alpha=0.5, label='Convergence step')
    par.set_ylabel('Convergence step')
    par.legend(loc='upper right')
    par.set_ylim(0, 5e5)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ax2 = host[1]
    ax2.errorbar(Nc_values, agg_data['steptime_mean'].to_numpy(), yerr=agg_data['steptime_std'], label='Time per step')

    ax2.set_xlabel('Nc')

    ax2.set_ylabel('Step time [s]')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, agg_data['steptime_mean'].max()*1.3)
    #ax.set_yscale('log')

    #ax.set_xlim(min(Nc_values) * 0.7, max(Nc_values) * 1.05)
    
    #fig.suptitle('Kullback-Leibler distance from converged distribution')
    #fig.suptitle('Convergence steps and time as a function of Nc')
    plt.show()


def convergence_tester(file_path):
    size = 5
    p_error = 0.15
    Nc = 19
    TOPS=20
    tops_burn=10
    steps=1000000

    # Number of times every parameter configuration is tested
    pop = 10

    #criteria = ['error_based', 'distr_based', 'majority_based', 'tvd_based', 'kld_based']
    criteria = ['error_based']
    crit = 'error_based'

    SEQ_list = [20, 25, 30, 35, 40]#[i for i in range(4, 25, 2)]

    eps_list = [2e-3*i for i in range(1, 6)]

    #crits_stats = {crit: [[], [], [], [], []] for crit in criteria}
    crits_stats = pd.DataFrame(columns=['SEQ', 'eps', 'kld', 'tvd', 'steps'])

    for SEQ in SEQ_list:
        for eps in eps_list:
            for j in range(pop):
                init_toric = Toric_code(size)
                init_toric.generate_random_error(p_error)

                [distr, eq, eq_full, chain0, burn_in, crits_distr] = parallel_tempering_analysis(init_toric, Nc, p=p_error, TOPS=TOPS, SEQ=SEQ, tops_burn=tops_burn, steps=steps, conv_criteria=criteria, eps=eps)

                distr = np.divide(distr.astype(np.float), 100)

                tvd_crit = tvd(distr, crits_distr[crit][0])
                kld_crit = kld(distr, crits_distr[crit][0])
                tmp_dict = {'SEQ': SEQ, 'eps': eps, 'kld': kld_crit, 'tvd': tvd_crit, 'steps': crits_distr[crit][1]}
                crits_stats = crits_stats.append(tmp_dict, ignore_index=True)

    crits_stats.to_pickle(file_path)


def conv_test_visuals(files=50):
    file_base = 'output/conv_data_'
    stats = pd.DataFrame(columns=['SEQ', 'eps', 'kld', 'tvd', 'steps'])
    for i in range(files):
        df = pd.read_pickle(file_base + str(i) + '.xz')
        stats = pd.concat([stats, df])

    agg_stats = ['SEQ', 'eps', 'nbr_converged', 'kld_mean', 'kld_std', 'kld_geom_mean', 'kld_geom_std', 'tvd_mean', 'tvd_std', 'tvd_geom_mean', 'tvd_geom_std', 'steps_mean', 'steps_std', 'steps_geom_mean', 'steps_geom_std']

    agg_data = pd.DataFrame(columns = agg_stats)

    SEQ_values = np.unique(stats['SEQ'].to_numpy())
    eps_values = np.unique(stats['eps'].to_numpy())

    # Number of data points
    tot_pts = stats.shape[0]
    # Number of different SEQ values
    SEQ_pts = SEQ_values.size
    # Number of unique eps values
    eps_pts = eps_values.size
    # Number of samples per (SEQ, eps) pair
    pop = int(tot_pts / (SEQ_pts * eps_pts))

    for SEQ in SEQ_values:
        for eps in eps_values:
            # Window of points for current (SEQ, eps) pair
            window = stats[(stats['SEQ'] == SEQ) & (stats['eps'] == eps)]
            # remove non converged runs
            window = window[window['steps'] != -1]
            # Mean and std values over converged runs
            agg = window.agg(['mean', 'std', geom_mean, geom_std, 'max', 'min'])
            # number of converged runs
            nbr_converged = window.shape[0]
            tmp_dict = {'SEQ': SEQ, 'eps': eps, 'nbr_converged': nbr_converged}
            for name in ['kld', 'tvd', 'steps']:
                for op in ['mean', 'std', 'geom_mean', 'geom_std', 'max', 'min']:
                    tmp_dict[name + '_' + op] = agg.loc[op][name]

            agg_data = agg_data.append(tmp_dict, ignore_index=True)
        
    plot_rows = int(np.ceil(np.sqrt(SEQ_pts)))
    plot_cols = int(np.ceil(SEQ_pts / plot_rows))
    
    #fig, axs = plt.subplots(plot_rows, plot_cols, constrained_layout=True)
    fig, host = plt.subplots(plot_rows, plot_cols)

    for i, SEQ in enumerate(SEQ_values):

        window = agg_data[agg_data['SEQ'] == SEQ]

        nbr_converged = window['nbr_converged']
        epss = window['eps']

        mean = 'arit'
        if mean == 'arit':
            klds = window['kld_mean']
            tvds = window['tvd_mean']
            kld_stds = window['kld_std']
            tvd_stds = window['tvd_std']
            steps = window['steps_mean']
            yerr = tvd_stds
            scale = 'linear'
        elif mean == 'geom':
            klds = window['kld_geom_mean']
            tvds = window['tvd_geom_mean']
            kld_stds = window['kld_geom_std']
            tvd_stds = window['tvd_geom_std']
            steps = window['steps_geom_mean']
            yerr = [tvds * (1 - 1/tvd_stds), klds * (tvd_stds - 1)]
            scale = 'log'
        
        tvd_min = window['tvd_min']
        tvd_max = window['tvd_max']

        row = i // plot_cols
        col = i % plot_cols

        ax = host[row][col]
        ax.set_zorder(2)
        ax.patch.set_visible(False)

        #ax.errorbar(epss, tvds, yerr=yerr, label='Distance')
        ax.plot(epss, tvds, label='Mean distance')
        #ax.plot(epss, tvd_min, label='Min distance')
        #ax.plot(epss, tvd_max, label='Max distance')

        ax.set_title('SEQ: ' + str(SEQ))
        ax.set_xlabel('eps')
        ax.set_ylabel('Distance')
        ax.legend(loc='upper left')
        #ax.set_yscale('log')
        ax.set_ylim(0, 0.03)
        #ax.set_yscale(scale)
        ax.set_xlim(min(epss) * 0.7, max(epss) * 1.05)

        par = ax.twinx()
        #par.bar(epss.to_numpy(), nbr_converged.to_numpy(), width=8e-4, color='gray', alpha=0.5, label='Converged samples')
        par.bar(epss.to_numpy(), steps.to_numpy(), width=8e-4, color='gray', alpha=0.5, label='Converged samples')
        par.set_ylabel('Convergence step')
        #par.legend(loc='upper right')
        par.set_ylim(0, 500000)
            
    #fig.suptitle('Kullback-Leibler distance from converged distribution')
    fig.suptitle('Total variational distance from converged distribution')

    ax = host[2][1]
    SEQ = 25
    eps = 0.008
    window = stats[(stats['SEQ'] == SEQ) & (stats['eps'] == eps)]
    window = window[window['steps'] != -1]
    #window = stats[stats['steps'] != -1]
    y = window['tvd'].sort_values()
    y = np.log(y)
    ax.hist(y, bins=25, density=True, range=(-6, -2))
    #ax.scatter(np.arange(window.shape[0]), y)
    #ax.axhline(y.mean())

    SEQ = 35
    eps = 0.002
    window = stats[(stats['SEQ'] == SEQ) & (stats['eps'] == eps)]
    window = window[window['steps'] != -1]
    #window = stats[stats['steps'] != -1]
    y = window['tvd'].sort_values()
    y = np.log(y)
    ax.hist(y, bins=25, density=True, alpha=0.5, range=(-6, -2))
    #ax.scatter(np.arange(window.shape[0]), y)
    #ax.axhline(y.mean())

    #ax.set_yscale('log')
    #ax.set_ylim(2e-3, 5e-1)

    plt.show()


def conv_stats(file_path, p_error, SEQ=30, eps=0.006):
    size = 5
    Nc = 19
    TOPS = 20
    tops_burn = 10
    steps = 1000000

    # Number of times every parameter configuration is tested
    pop = 200

    file_path = file_path.format(SEQ, eps).replace('0.', '0') + '.xz'

    #criteria = ['error_based', 'distr_based', 'majority_based', 'tvd_based', 'kld_based']
    criteria = ['error_based']
    crit = 'error_based'

    #crits_stats = {crit: [[], [], [], [], []] for crit in criteria}
    crits_stats = pd.DataFrame(columns=['kld', 'tvd', 'steps', 'success', 'distr'])

    for j in range(pop):
        init_toric = Toric_code(size)
        init_toric.generate_random_error(p_error)

        seed_class = define_equivalence_class(init_toric.qubit_matrix)

        [distr, eq, eq_full, chain0, burn_in, crits_distr] = parallel_tempering_analysis(init_toric, Nc, p=p_error, TOPS=TOPS, SEQ=SEQ, tops_burn=tops_burn, steps=steps, conv_criteria=criteria, eps=eps)

        distr = np.divide(distr.astype(np.float), 100)

        distr_conv = crits_distr[crit][0]
        tvd_crit = tvd(distr, distr_conv)
        kld_crit = kld(distr, distr_conv)
        
        # seed is succesfully corrected if most likely class is the same as the seed class
        success = (np.argmax(distr_conv) == seed_class)

        tmp_dict = {'kld': kld_crit, 'tvd': tvd_crit, 'steps': crits_distr[crit][1], 'success': success, 'distr': distr_conv}
        crits_stats = crits_stats.append(tmp_dict, ignore_index=True)

        if not j % 10:
            crits_stats.to_pickle(file_path)

    crits_stats.to_pickle(file_path)


def conv_stats_visuals(files=50, SEQ=30, eps=0.006, p=0.1):
    file_base = 'output/conv_stats_' + 'SEQ{}_eps{:.3f}_p{:.3f}_'.format(SEQ, eps, p)
    stats = pd.DataFrame(columns=['kld', 'tvd', 'steps', 'success', 'distr'])
    for i in range(files):
        filename = (file_base + str(i)).replace('0.', '0')  + '.xz'
        df = pd.read_pickle(filename)
        stats = pd.concat([stats, df])

    # remove non converged runs
    converged = stats[stats['steps'] != -1]
    
    y = converged['tvd']
    #y = y[y > 0]
    print(converged[y <= 0])
    #y = np.log(y)
    plt.hist(y, bins=500, density=True, range=(0, 0.05))
    plt.show()

    success_rate = stats['success'].mean()
    distrs = converged['distr'].to_numpy()
    p_max_mean = np.mean([np.max(distr) for distr in distrs])
    nbr_converged = converged.shape[0]
    tot_pts = stats.shape[0]

    print('Success rate:', success_rate)
    print('Mean of p_max:', p_max_mean)
    print('Total runs:', tot_pts)
    print('Converged runs:', nbr_converged)


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


if __name__ == '__main__':
    try:
        array_id = str(sys.argv[1])
        local_dir = str(sys.argv[2])
        p_error = float(sys.argv[3])
    except:
        array_id = '0'
        local_dir = '.'
        p_error = 0.1
    
    #  Build file path
    #file_path = os.path.join(local_dir, 'Nc_data_' + array_id + '.xz')
    #Nc_tester(file_path=file_path, Nc_interval=[3, 31])

    #file_path = os.path.join(local_dir, 'conv_data_' + array_id + '.xz')
    #convergence_test(file_path)

    #file_path = os.path.join(local_dir, 'conv_stats_p' + str(p_error).replace('.', '') + '_' + array_id + '.xz')
    file_path = os.path.join(local_dir, 'conv_stats_SEQ{}_eps{:.3f}_' + 'p{:.3f}_{}'.format(p_error, array_id))
    conv_stats(file_path, p_error)
    #conv_stats_visuals()

    #Nc_visuals(files = 6)
    #conv_test_visuals()
