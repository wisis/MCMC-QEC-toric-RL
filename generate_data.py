from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy
import pandas as pd
import os
from termcolor import colored
import sys


class MCMCDataReader:  # This is the object we crate to read a file during training
    def __init__(self, file_path, size, df_name='df'):  # lägg till en df för metadata? typ size, isf ha HDF?
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
            return None, None  # How do we do this nicely?
    
    def has_next(self):
        return self.__current_index < self.__capacity

    def current_index(self):
        return self.__current_index

# some functionality to stop mid run and save?
# https://stackoverflow.com/questions/7180914/pause-resume-a-python-script-in-middle

#@profile  # Efter att ha kört med profiler med 20 steps, 10 iters (to little hehe) tar ändå parallell tempering 97% av tiden
def generate(file_path, max_capacity=10000, nbr_datapoints=10, df_name='df'):
    # All paramteters for data generation is set here, some may be irrelevant depending on the choice of others
    params = {  'size':5,
                'p':0.10,
                'Nc':11,
                'steps':20000,
                'iters':10,
                'conv_criteria':'distr_based',
                'SEQ':10,
                'TOPS':2,
                'eps':0.1}

    size = 5

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
        init_toric = Toric_code(size)
        init_toric.generate_random_error(params['p'])

        # generate data for DataFrame storage  OBS now using full bincount, change this
        [_, df_eq_distr, _, _] = parallel_tempering(init_toric, size, p=0.10, steps=params['steps'], iters=10, conv_criteria='distr_based')
        df_qubit = init_toric.qubit_matrix.reshape((-1))  # can also use flatten here?
        
        # create indices for generated data
        names = ['data_nr', 'layer', 'x', 'y']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(2), np.arange(size), np.arange(size)], names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(16)+2, [0], [0]], names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame(df_qubit.astype(np.uint16), index=index_qubit, columns=['data'])
        df_distr = pd.DataFrame(df_eq_distr.astype(np.uint16), index=index_distr, columns=['data']) # dtype for eq_distr? want uint16

        # Add dataframes to list, we do not append to df here because of O(L) time complexity
        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Add to df and save somewhat continuously ----------------------------

    if len(df_list) > 0:
        df = df.append(df_list)
        print(colored('\nSaving all generated data', 'green'))
        df.to_pickle(file_path)
    
    print(colored('\nCompleted', 'green'))


if __name__ == '__main__':
    file_path=os.path.join(os.getcwd(), "data", 'data.xz')
    generate(file_path, 10, 10)
    #view_all_data(file_path)
    iterator = MCMCDataReader(file_path, 5)
    while iterator.has_next():
        print(colored('Datapoint nr: '+ str(iterator.current_index() + 1), 'red'))
        print(iterator.next())
