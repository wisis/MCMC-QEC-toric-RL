from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv

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


def conv_test_visuals():
    file_name = 'conv_test_SEQ_2_10_p_0.15.xz'
    crits_stats = pd.read_pickle(file_name)

    criteria = ['error_based', 'distr_based', 'majority_based', 'tvd_based', 'kld_based']

    stats = ['criteria', 'SEQ', 'tol', 'kld', 'tvd', 'steps']
    agg_stats = ['criteria', 'SEQ', 'tol', 'nbr_converged', 'kld', 'std_kld', 'tvd', 'std_tvd', 'steps', 'std_steps']

    crit = 'error_based'

    data = crits_stats[crits_stats['criteria'] == crit]
    agg_data = pd.DataFrame(columns = agg_stats)

    # Number of data points
    tot_pts = data.shape[0]
    # Number of different SEQ values
    SEQ_pts = data['SEQ'].value_counts().size
    # Number of different tol values
    tol_pts = data['tol'].value_counts().size
    # Number of unique (SEQ, tol) pairs
    configurations = SEQ_pts * tol_pts
    # Number of samples per (SEQ, tol) pair
    pop = int(tot_pts / configurations)

    def geom_mean(series):
        array=series.to_numpy()
        return np.exp(np.average(np.log(array)))

    def geom_std(series):
        array=series.to_numpy()
        return np.exp(np.std(np.log(array)))


    for i in range(configurations):
        # Window of points for current (SEQ, tol) pair
        window = data[i * pop : (i + 1) * pop]
        # Bool series of converged runs
        converged = window['steps'] != -1
        # Mean and std values over converged runs
        agg = window[converged].agg(['mean', 'std', geom_mean, geom_std])
        # number of converged runs
        nbr_converged = window[converged].shape[0]
        tmp_dict = {'criteria': crit, 'SEQ': window['SEQ'][i * pop], 'tol': window['tol'][i * pop], 'nbr_converged': nbr_converged}
        for name in ['kld', 'tvd', 'steps']:
            tmp_dict[name] = agg.loc['geom_mean'][name]
            tmp_dict['std_' + name] = agg.loc['geom_std'][name]

        agg_data = agg_data.append(tmp_dict, ignore_index=True)
        
    plot_rows = int(np.ceil(np.sqrt(SEQ_pts)))
    plot_cols = int(np.ceil(SEQ_pts / plot_rows))
    
    #fig, axs = plt.subplots(plot_rows, plot_cols, constrained_layout=True)
    fig, host = plt.subplots(plot_rows, plot_cols)

    for i in range(SEQ_pts):
        #title = 'SEQ: ' + str(agg_data['SEQ'][i * tol_pts])
        title = 'SEQ: ' + str(data['SEQ'][i * tol_pts * pop])
        xlabel = 'Tolerance'

        #window = agg_data[i * tol_pts: (i + 1) * tol_pts]
        window = data[i * tol_pts * pop: (i + 1) * tol_pts * pop]

        converged = window['steps'] != -1
        tols = window['tol'][converged]
        klds = window['kld'][converged]
        tvds = window['tvd'][converged]

        #nbr_converged = window['nbr_converged']
        #std_klds = window['std_kld']
        #std_tvds = window['std_tvd']
        nbr_converged = agg_data['nbr_converged'][i * tol_pts: (i + 1) * tol_pts]

        row = i // plot_cols
        col = i % plot_cols

        #ax = fig.add_subplot(plot_rows, plot_cols, i + 1)
        ax = host[row][col]
        par = ax.twinx()
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        #ax = axs[row, col]
        #yerr = [klds * (1 - 1/std_tvds), klds * (std_tvds - 1)]
        #ax.errorbar(tols, klds, yerr=yerr, label='Distance')
        ax.scatter(tols, tvds, label='Distance')

        #par.bar(tols.to_numpy(), nbr_converged.to_numpy(), width=8e-4, color='gray', alpha=0.5, label='Converged samples')
        par.bar(window['tol'][0::pop].to_numpy(), nbr_converged.to_numpy(), width=8e-4, color='gray', alpha=0.5, label='Converged samples')

        ax.set_title(title)
        #ax.set_xlabel(xlabel)
        #ax.set_xscale('log')
        ax.set_ylabel('Distance')
        ax.legend(loc='lower left')
        par.set_ylabel('Converged samples')
        par.legend(loc='lower right')
        ax.set_ylim(1e-3, 1e0)
        ax.set_yscale('log')
        
        ax.set_xlim(min(tols) * 0.7, max(tols) * 1.05)
            
    #fig.suptitle('Kullback-Leibler distance from converged distribution')
    fig.suptitle('Total variational distance from converged distribution')
    plt.show()

if __name__ == '__main__':
    convergence_analysis()