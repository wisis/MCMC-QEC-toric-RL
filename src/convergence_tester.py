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

from matplotlib import rc
#rc('font',**{'family':'sans-serif'})#,'sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif'})#,'serif':['Palatino']})
rc('text', usetex=True)


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


def Nc_visuals(files=6, SEQ=20):
    file_base = 'output/Nc_data_SEQ{}_'.format(SEQ)

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

    stats['steps'].replace(-1, 1e6, inplace=True)

    for Nc in Nc_values:
        # Window of points for current Nc value
        window = stats[stats['Nc'] == Nc].copy()
        # Only look at converged runs
        #window['steps'] = window['steps'].replace(-1, 1e6)
        # Calculate step time
        window['steptime'] = window['time'] / window['steps']
        # Mean and std values over converged runs
        agg = window.agg(['mean', 'std', geom_mean, geom_std])
        # number of converged runs
        nbr_converged = window[window['steps'] != 1e6].shape[0]
        # temporary dict to append to aggregated data
        tmp_dict = {'Nc': Nc, 'nbr_converged': nbr_converged}
        for name in ['time', 'steps', 'steptime']:
            for op in ['mean', 'std', 'geom_mean', 'geom_std']:
                tmp_dict[name + '_' + op] = agg.loc[op][name]

        # append aggregate data for current Nc value to agg_data
        agg_data = agg_data.append(tmp_dict, ignore_index=True)
    
    #stats.boxplot(column='time', by='Nc', return_type='both')

    time = agg_data['time_mean'].to_numpy()
    time_err = agg_data['time_std'] / np.sqrt(pop)

    left = 0.12
    plt.rcParams.update({'font.size': 48, 'figure.subplot.top': 0.91, 'figure.subplot.bottom': 0.18, 'figure.subplot.left': left, 'figure.subplot.right': 1 - left})
    plt.rc('axes', labelsize=60)#, titlesize=40) 

    c = plt.cm.cubehelix(0.2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(Nc_values, time, yerr=time_err, fmt='-', ms=24, lw=6, capsize=12, capthick=6, c=c)#, label='Convergence time')

    pad = 20.0
    ax.set_xlabel('\\texttt{Nc}', labelpad=pad)
    ax.set_ylabel('Konvergenstid [s]', labelpad=pad)
    ax.set_ylim([0, 110])
    ax.set_xlim([2, 32])
    ax.set_title('SEQ = {}'.format(SEQ), pad=pad)
    ax.grid(True, axis='both')
    #ax.legend(loc='upper right')
    #plt.savefig('plots/Nc_data_SEQ{}.png'.format(SEQ))
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


def conv_test_visuals():
    files = 100
    file_base = 'output/conv_data_'
    stats = pd.DataFrame(columns=['SEQ', 'eps', 'kld', 'tvd', 'steps'])
    for i in range(files):
        df = pd.read_pickle(file_base + str(i) + '.xz')
        stats = pd.concat([stats, df])

    agg_stats = ['SEQ', 'eps', 'nbr_converged', 'kld_mean', 'kld_std', 'kld_geom_mean', 'kld_geom_std', 'tvd_mean', 'tvd_std', 'tvd_geom_mean', 'tvd_geom_std', 'steps_mean', 'steps_std', 'steps_geom_mean', 'steps_geom_std']

    agg_data = pd.DataFrame(columns = agg_stats)

    SEQ_values = np.unique(stats['SEQ'].to_numpy())
    eps_values = np.unique(stats['eps'].to_numpy())

    def weight(tvd, steps):
        return tvd

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

            tmp_dict['tvd_steps_outer'] = weight(tmp_dict['tvd_mean'], tmp_dict['steps_mean'])
            tmp_dict['tvd_steps_inner'] = weight(window['tvd'], window['steps']).mean()

            agg_data = agg_data.append(tmp_dict, ignore_index=True)
        
    plot_cols = int(np.ceil(np.sqrt(SEQ_pts)))
    plot_rows = int(np.ceil(SEQ_pts / plot_cols))

    left = 0.07
    plt.rcParams.update({'figure.subplot.top': 0.97, 'figure.subplot.bottom': 0.12, 'figure.subplot.left': left, 'figure.subplot.right': 1 - left})
    #fig, axs = plt.subplots(plot_rows, plot_cols, constrained_layout=True)
    #fig, host = plt.subplots(plot_rows, plot_cols)
    fig = plt.figure()

    #locs = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3)]
    #host = [plt.subplot2grid((plot_rows, plot_cols * 2), loc, colspan = 2) for loc in locs]

    ax = fig.add_subplot(111)

    big_font = 30
    small_font = 24

    linestyles = ['-', '--', '-.', (0, (3, 1, 3, 1, 1, 1)), (0, (2, 1))]
    color=iter(plt.cm.cubehelix(np.linspace(0.2, 0.7, SEQ_values.size)))

    for i, SEQ in enumerate(SEQ_values):
        c = next(color)

        window = agg_data[agg_data['SEQ'] == SEQ]

        nbr_converged = window['nbr_converged']
        epss = window['eps']
        
        tvds = window['tvd_mean']
        tvd_stds = window['tvd_std']

        steps = window['steps_mean']

        tvd_min = window['tvd_min']
        tvd_max = window['tvd_max']

        y = window['tvd_steps_inner']

        #row = i // plot_cols
        #col = i % plot_cols
        #ax = host[row][col]
        #ax.set_zorder(2)
        #ax.patch.set_visible(False)

        #ax = host[i]
        ls = linestyles[i]
        #ax.errorbar(epss, tvds, yerr=tvd_stds, label='Distance')
        ax.plot(epss, y, label='SEQ = {}'.format(SEQ), lw = 4, ls = ls, c = c)
        #ax.set_yscale('log')
        #ax.plot(epss, tvd_min, label='Min distance')
        #ax.plot(epss, tvd_max, label='Max distance')

    if True:
        pad = 20.0
        #ax.set_title('SEQ: ' + str(SEQ), fontsize = title_fontsize)
        ax.set_xlabel('\\texttt{eps}', fontsize=big_font, labelpad=pad)
        ax.set_ylabel('Maximal distans', fontsize=big_font, labelpad=pad)
        #ax.legend(loc='upper left')
        #ax.set_yscale('log')
        #ax.set_ylim(0.01, 0.025)
        #ax.set_yscale(scale)
        #ax.set_xlim(min(epss) * 0.7, max(epss) * 1.05)
        ax.grid(True, axis='both')
        ax.tick_params(axis='both', which='major', labelsize=small_font)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, -2))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, -3))
        ax.yaxis.offsetText.set_fontsize(small_font)
        ax.xaxis.offsetText.set_fontsize(small_font)

        '''
        par = ax.twinx()
        #par.bar(epss.to_numpy(), nbr_converged.to_numpy(), width=8e-4, color='gray', alpha=0.5, label='Converged samples')
        par.bar(epss.to_numpy(), steps.to_numpy(), width=8e-4, color='gray', alpha=0.5, label='Converged samples')
        par.set_ylabel('Convergence step')
        #par.legend(loc='upper right')
        par.set_ylim(0, 500000)
        '''
    
    ax.legend(loc='upper left', fontsize=small_font, handlelength=2.5)
    #fig.suptitle('Kullback-Leibler distance from converged distribution')
    #fig.suptitle('Total variational distance from converged distribution')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.set_cmap('hot') 
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    Y = SEQ_values
    X = eps_values
    Z = np.reshape(agg_data['tvd_steps_inner'].to_numpy(), (eps_pts, SEQ_pts))

    cont = ax2.contourf(X, Y, Z, cmap='cubehelix')
    plt.colorbar(cont, ax=ax2)

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
    
    end = 0.06 if p == 0.1 else 0.08

    file_base = 'output/conv_stats_' + 'SEQ{}_eps{:.3f}_p{:.3f}_'.format(SEQ, eps, p)
    stats = pd.DataFrame(columns=['kld', 'tvd', 'steps', 'success', 'distr'])
    for i in range(files):
        filename = (file_base + str(i)).replace('0.', '0')  + '.xz'
        df = pd.read_pickle(filename)
        stats = pd.concat([stats, df])

    # remove non converged runs
    converged = stats[stats['steps'] != -1]
    
    y = converged['tvd']

    y = y[y != 0]
    #y = y[y > 0]
    #y = np.log(y)
    #print(y[(y > 0.0099) & (y < 0.01)])

    converged = converged[converged['tvd'] != 0]
    success_rate = converged['success'].mean()
    distrs = converged['distr'].to_numpy()
    p_max_mean = np.mean([np.max(distr) for distr in distrs])
    nbr_converged = converged.shape[0]
    tot_pts = stats.shape[0]

    perc95 = np.percentile(y, 95)
    print('95th percentile:', perc95)
    print('Success rate:', success_rate)
    print('Mean of p_max:', p_max_mean)
    print('Total runs:', tot_pts)
    print('Converged runs:', nbr_converged)

    small_font = 48
    mid_font = 60

    left = 0.12
    plt.rcParams.update({'figure.subplot.top': 0.98, 'figure.subplot.bottom': 0.16, 'figure.subplot.left': left, 'figure.subplot.right': 1 - left})

    c = plt.cm.cubehelix(0.2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(y, bins=100, density=True, range=(0, end), color=c)
    ax.axvline(x = perc95, ls='--', lw=8, c='k', label=('$P_{95}$ = ' + '{:.3f}'.format(perc95)))
    
    pad = 20.0
    ax.set_xlabel('Maximal distans', fontsize = mid_font, labelpad = pad)
    ax.set_ylabel('Antal', fontsize = mid_font, labelpad = pad)
    ax.tick_params(axis='both', which='major', labelsize=small_font)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, -2))
    ax.xaxis.offsetText.set_fontsize(small_font)
    ax.legend(loc = 'upper right', fontsize = mid_font, handlelength=1)
    ax.grid(True, axis='both')

    plt.show()


def bias_tester(file_path, p_error, SEQ=30, eps=0.006):
    size = 5
    Nc = 19
    TOPS = 20
    tops_burn = 10
    steps = 1000000

    # Number of times every parameter configuration is tested
    pop = 200

    file_path = file_path.format(SEQ, eps).replace('0.', '0') + '.xz'

    # Choose convergence criteria to test. Never did any other than error_based
    crit = 'error_based'

    # Create dataframe to hold results
    crits_stats = pd.DataFrame(columns=['kld', 'tvd', 'steps', 'success', 'distr'])

    for j in range(pop):
        # Generate seed
        init_toric = Toric_code(size)
        init_toric.generate_random_error(p_error)
        
        # Create uniformly sampled chain (morph) with same syndrome as seed
        init_error_morph, _ = apply_random_logical(init_toric.qubit_matrix)
        init_error_morph = apply_stabilizers_uniform(init_error_morph)

        # Calculate equivalence classes of seed and morph
        class_seed = define_equivalence_class(init_toric.qubit_matrix)
        class_morph = define_equivalence_class(init_error_morph)

        # Run MCMC on seed
        [distr_seed, steps_seed] = parallel_tempering_plus(init_toric, Nc, p=p_error, TOPS=TOPS, SEQ=SEQ, tops_burn=tops_burn, steps=steps, conv_criteria=crit, eps=eps)
        distr_seed = np.divide(distr_seed.astype(np.float), 100)


        # Run MCMC on morph
        init_toric.qubit_matrix = init_error_morph
        [distr_morph, steps_morph] = parallel_tempering_plus(init_toric, Nc, p=p_error, TOPS=TOPS, SEQ=SEQ, tops_burn=tops_burn, steps=steps, conv_criteria=crit, eps=eps)
        distr_morph = np.divide(distr_morph.astype(np.float), 100)

        # Use a dictionary for easy way to append the dataframe
        tmp_dict = {'steps_seed': steps_seed, 'class_seed': class_seed, 'distr_seed': distr_seed, 
                    'steps_morph': steps_morph, 'class_morph': class_morph, 'distr_morph': distr_morph}

        crits_stats = crits_stats.append(tmp_dict, ignore_index=True)

        if not j % 10:
            crits_stats.to_pickle(file_path)

    crits_stats.to_pickle(file_path)


def bias_stats(p):
    p_str = 'p{:.3f}'.format(p).replace('.', '')
    print(p_str)

    df = pd.DataFrame()

    drop = ['kld', 'tvd', 'steps', 'success', 'distr']

    for file in os.listdir('output/'):
        if file.startswith('bias') and p_str in file:
            tmp = pd.read_pickle('output/' + file)
            tmp.drop(drop, axis=1, inplace=True)
            df = pd.concat([df, tmp], ignore_index=True)
    
    for init in ['seed', 'morph']:

        z = 1.96 # 95% confidence

        conv = df.copy()  #[df['steps_' + init] != -1]
        conv['steps_' + init].replace(-1.0, 1e6, inplace=True)
        print('Converged {}: {}'.format(init, df[df['steps_' + init] != -1].shape[0]))

        max_p = conv['distr_' + init].apply(lambda x: np.max(x))
        mean_max_p = max_p.mean()
        std_max_p = max_p.std()
        conf_max_p = std_max_p / np.sqrt(conv.shape[0]) * z
        print('Mean max p {}: {:.4f} +- {:.4f}'.format(init, mean_max_p, conf_max_p))

        success = (conv['class_seed'] == conv['distr_' + init].apply(lambda x: np.argmax(x)))
        P_s = success.mean()
        std_P_s = success.std()
        #conf_P_s = z * np.sqrt( (P_s * (1 - P_s)) / conv.shape[0] )
        conf_P_s = std_P_s / np.sqrt(conv.shape[0]) * z
        print('Success rate {}: {} +- {:.4f}'.format(init, P_s, conf_P_s))
        #print(std_P_s / np.sqrt(conv.shape[0]) * z)

        mean_steps = conv['steps_' + init].mean()
        std_steps = conv['steps_' + init].std()
        print('Mean steps {}: {:.0f} +- {:.0f}'.format(init, mean_steps, std_steps))    

    print('Non converged both: {}'.format(df[(df['steps_seed'] == -1) & (df['steps_morph'] == -1)].shape[0]))
    plt.show()


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

    #file_path = os.path.join(local_dir, 'conv_stats_SEQ{}_eps{:.3f}_' + 'p{:.3f}_{}'.format(p_error, array_id))
    #conv_stats(file_path, p_error)
    #file_path = os.path.join(local_dir, 'bias_test_SEQ{}_eps{:.3f}_' + 'p{:.3f}_{}'.format(p_error, array_id))
    #bias_tester(file_path, p_error)

    #Nc_visuals(files = 6, SEQ = 8)
    conv_test_visuals()
    #conv_stats_visuals(p=0.185)

    #bias_stats(p=0.1)
