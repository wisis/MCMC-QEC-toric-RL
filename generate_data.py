from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd
import os
from termcolor import colored
import sys


class MCMCDataReader:  # This is the object we crate to read a file during training
    def __init__(self, file_path, size):
        self.__file_path = file_path
        self.__size = size
        self.__df = pd.read_pickle(file_path)
        self.__current_index = 0
        self.__capacity = self.__df.index[-1][0] + 1  # The number of data samples in the dataset

    def next(self):
        if self.__current_index < self.__capacity:
            qubit_matrix = self.__df.loc[self.__current_index, 0:1, :, :].to_numpy(copy=True).reshape((2, self.__size, self.__size))
            eq_distr = self.__df.loc[self.__current_index, 2:17, 0, 0].to_numpy(copy=True).reshape((-1))  # kanske inte behöver kopiera här?
            self.__current_index += 1
            return qubit_matrix, eq_distr
        else:
            return None, None  # How do we do this nicely? Maybe it can wrap around?
    
    def has_next(self):
        return self.__current_index < self.__capacity

    def current_index(self):
        return self.__current_index


#@profile  # Efter att ha kört med profiler med 20 steps, 10 iters (to little hehe) tar ändå parallell tempering 97% av tiden
def generate(file_path, params, max_capacity=10000, nbr_datapoints=100000000):
    # Vi behöver öppna filen och kolla hur många entrys som finns
    try:
        df = pd.read_pickle(file_path)
        nbr_existing_data = df.index[-1][0] + 1
    except:
        df = pd.DataFrame()
        nbr_existing_data = 0

    print(colored('\nDataFrame with ' + str(nbr_existing_data) + ' datapoints opened at: ' + str(file_path), 'blue'))

    # Stop the file from exceeding the max limit nbr of datapoints
    nbr_to_generate = min(max_capacity-nbr_existing_data, nbr_datapoints)
    if nbr_to_generate < nbr_datapoints:
        print('Generating ' + str(max(nbr_to_generate, 0)) + ' datapoins instead of ' + str(nbr_datapoints) + ', as the given number would overflow existing file')

    df_list = []
    for i in np.arange(nbr_to_generate) + nbr_existing_data:
        print('Starting generation of point nr: ' + str(i + 1))
        
        # Initiate toric
        init_toric = Toric_code(params['size'])
        init_toric.generate_random_error(params['p'])

        # generate data for DataFrame storage  OBS now using full bincount, change this
        [df_eq_distr, _, _, _, _] = parallel_tempering(init_toric, params['Nc'],p=params['p'], steps=params['steps'],
                                                                iters=params['iters'], conv_criteria=params['conv_criteria'])
        
        df_qubit = init_toric.qubit_matrix.reshape((-1))  # flatten qubit matrix to store in dataframe
        
        # create indices for generated data
        names = ['data_nr', 'layer', 'x', 'y']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(2), np.arange(params['size']), np.arange(params['size'])], names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(16)+2, [0], [0]], names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame(df_qubit.astype(np.uint8), index=index_qubit, columns=['data'])
        df_distr = pd.DataFrame(df_eq_distr.astype(np.uint8), index=index_distr, columns=['data']) # dtype for eq_distr? want uint16

        # Add dataframes to list, we do not append to df here because of O(L) time complexity
        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Add to df and save somewhat continuously ----------------------------
        if (i + 1) % 1000 == 0:
            df = df.append(df_list)
            df_list.clear()
            print(colored('Intermediate save point reached (writing over)', 'green'))
            df.to_pickle(file_path)

    if len(df_list) > 0:
        df = df.append(df_list)
        print(colored('\nSaving all generated data (writing over)', 'green'))
        df.to_pickle(file_path)
    
    print(colored('\nCompleted', 'green'))


if __name__ == '__main__':
    # All paramteters for data generation is set here, some may be irrelevant depending on the choice of others
    params = {  'size':5,
                'p':0.10,
                'Nc':11,
                'steps':10000,
                'iters':10,
                'conv_criteria':'none',
                'SEQ':2,
                'TOPS':10,
                'eps':0.1}

    # get job array id
    try:
        array_id = str(sys.argv[1])
    except:
        array_id = '0003'

    # build file path
    file_path=os.path.join(os.getcwd(), "data", 'data_' + array_id + '.xz')
    
    # generate data
    generate(file_path, params, 15)
    
    #view_all_data(file_path)
    iterator = MCMCDataReader(file_path, params['size'])
    while iterator.has_next():
        print(colored('Datapoint nr: '+ str(iterator.current_index() + 1), 'red'))
        print(iterator.next())
