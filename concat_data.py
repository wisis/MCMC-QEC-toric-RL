import pandas as pd
import argparse
import pickle  # not used
import os


# Merges multiple data files into one
def concat_data(filename, nbr_files, extension):

    # Open first file and register number of data points
    first_file = os.path.join(filename + str(0) + extension)
    full_file = pd.read_pickle(first_file)
    full_file_index = full_file.index[-1][0]

    # List to hold file data until appended to new file
    file_list = []

    # Loop to merge files
    for i in range(nbr_files-1):

        current_filename = os.path.join(filename + str(i+1) + extension)
        current_file = pd.read_pickle(current_filename)
        current_file_index = current_file.index[-1][0] + 1

        # List to hold data points
        data_list = []

        for j in range(current_file_index):  # 0-9

            # Read each data point from column data_nr in the current file
            current_data_point = current_file.iloc[
                current_file.index.get_level_values('data_nr') == j]

            # Reset index, lift index col data_nr to column to replace values
            current_data_point = current_data_point.reset_index('data_nr')

            # (Suggestion) Change index for the current file to match full file index
            # |
            # V
            # Renaming data_nr index for the current data point according to full file index counter
            current_data_point['data_nr'] = (full_file_index + 1 + j)

            # Set index, lower col data_nr to multi index
            # to be used as index again
            current_data_point.set_index('data_nr', append=True,
                                         drop=True, inplace=True)
            current_data_point = current_data_point.reorder_levels(['data_nr',
                                                                    'layer',
                                                                    'x',
                                                                    'y'])

            # Append current data point to data_list which holds
            # the data points (with new updated data_nr index) temporarily
            data_list.append(current_data_point)

        # Updating full file index counter after each individual file
        full_file_index += current_file_index
        # Adding elements of data_list to file_list (list extends another list)
        file_list.extend(data_list)
        # Progress print
        print("Concatenating file " + str(i+2) + " out of "
              + str(nbr_files) + " to original.")

    # Append to full_file,
    # if appending data point per data point add data_list here
    full_file = full_file.append(file_list)

    # Create new file path for the new file
    filepath = os.path.join(filename + str("_concat") + extension)

    # Save concatenated file to new file
    full_file.to_pickle(filepath)

    print("Concatenation done. \nNew file with " + str(full_file_index + 1)
          + " data points can be found at: " + filepath)


if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(
        description='Concatenate compressed MCMC data files')

    # Create arguments for parser
    parser.add_argument(
        "filename_root",
        help="input filename root for all files you wish to concatenate",
        type=str)

    parser.add_argument("nbr_files", help="integer number of files", type=int)
    parser.add_argument("file_extension",
                        help="file extension for the files to assist in parsing",
                        type=str)

    # Parse arguments
    args = parser.parse_args()

    filename = args.filename_root
    nbr = args.nbr_files
    extension = args.file_extension

    if filename and nbr and extension:
        print("Congatenating", nbr, "files...")
        concat_data(filename, nbr, extension)
    else:
        print("Some arguments missing.")
