
import pandas as pd
import argparse
import pickle
import os

def concat_data(filename, nbr_files, extension):
    # open first file and nbr of data points
    first_file = os.path.join(filename + str(0) + extension)
    full_file = pd.read_pickle(first_file)
    full_file_index = full_file.index[-1][0]

    # list to hold file data until appended to new file
    file_list = []
    for i in range(nbr_files-1):
        current_filename = os.path.join(filename + str(i+1) + extension)
        current_file = pd.read_pickle(current_filename)
        current_file_index = current_file.index[-1][0] + 1

        # list to hold data points
        data_list = []
        for j in range(current_file_index): #0-9
            # read each data point from column data_nr in the current file
            current_data_point = current_file.iloc[current_file.index.get_level_values('data_nr') == j]

            # reset index, lift index col data_nr to column to replace values
            current_data_point = current_data_point.reset_index('data_nr')

            # renaming data_nr index for the current data point according to full file index counter
            current_data_point['data_nr'] = (full_file_index + 1 + j)

            # set index, lower col data_nr to multi index to be used as index again
            current_data_point.set_index('data_nr', append=True, drop=True, inplace=True)
            current_data_point = current_data_point.reorder_levels(['data_nr', 'layer', 'x', 'y'])

            # append current data point to data_list which holds the data points (with new updated data_nr index) temporarily
            data_list.append(current_data_point)

        # updating full file index counter after each individual file
        full_file_index += current_file_index
        # adding elements of data_list to file_list (list extends another list)
        file_list.extend(data_list)
        # progress print
        print("Concatenating file " + str(i+2) + " out of " + str(nbr_files) + " to original.")

    # append to full_file, if appending data point per data point add data_list here
    full_file = full_file.append(file_list)

    # create new file path for the new file
    filepath = os.path.join(filename + str("_concat") + extension)

    # save concatenated file to new file
    full_file.to_pickle(filepath)

    print("Concatenation done. \nNew file with " + str(full_file_index + 1) + " data points can be found at: " + filepath)


if __name__ == '__main__':
    # create parser
    parser = argparse.ArgumentParser(description='Concatenate compressed MCMC data files')

    # create arguments for parser
    parser.add_argument("filename_root", help="input filename root for all files you wish to concatenate", type=str)
    parser.add_argument("nbr_files", help="integer number of files", type=int)
    parser.add_argument("file_extension", help="file extension for he files to assist in parsing", type=str)

    # parse arguments
    args = parser.parse_args()

    filename = args.filename_root
    nbr = args.nbr_files
    extension = args.file_extension

    if filename and nbr and extension:
        print("Congatenating", nbr, "files..." )
        concat_data(filename, nbr, extension)
    else:
        print("Some arguments missing.")
