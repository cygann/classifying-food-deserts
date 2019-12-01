import math
import os
import numpy as np
from tqdm import trange
from tqdm import tqdm
import pickle
import random

path_to_script = os.path.dirname(os.path.abspath(__file__))
FULL_DATA_PICKLE = os.path.join(path_to_script, "data_sample_final.pickle")
data_files = [os.path.join(path_to_script, "data_parts/data_i.pickle"),
        os.path.join(path_to_script, "data_parts/data_ii.pickle"),
        os.path.join(path_to_script, "data_parts/data_iii.pickle")]
input_data = os.path.join(path_to_script, "data_sample.pickle")

"""
Since data was processed in multiple parts on myth machines, the three pickle
files need to be combined into one pickle file. This file (particularly this
function) does just that. This function will open the three .pickle files
specified in data_files and merges all the dicts into one dict, returning it.
"""
def coalese_data_files():
    merged_data = {}

    for p in data_files:
        data = None
        with open(p, 'rb') as fp:
            data = pickle.load(fp)

        zipcodes = list(data.keys())
        for z in zipcodes:
            info = data[z]
            merged_data[z] = info

    return merged_data
    
"""
Reads in a dict of zipcodes to (feature matrix, label) tuples and turns the
feature matrix into a feature vector of size 20, where the first 10 features
are the 2015 values and the last 10 are the percent changes from 2012 to 2015.
This dict is returned by the function.
"""
def build_percent_change_features():
    data = None
    with open(input_data, 'rb') as fp:
        data = pickle.load(fp)

    keys = list(data.keys())
    final_data = {}

    for zipcode in keys:
        point = data[zipcode]
        data_matrix = point[0]
        label = point[1]

        # Take second column, which represents 2015.
        data_2015 = data_matrix[:, 1]
        # Take first column, which represents 2012.
        data_2012 = data_matrix[:, 0]
        diff = data_2015 - data_2012
        percent = diff / data_2012 # Get the percent change.

        feature_vec = np.concatenate((data_2015, percent))

        final_data[zipcode] = (feature_vec, label)

    return final_data

def main():
    data = build_percent_change_features()
    print('Successfully read in a dataset of', len(data), 'datapoints.')

    # Save full data to pickle.
    with open(FULL_DATA_PICKLE, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Successfully merged all paritions of zip code data.")

if __name__== "__main__":
  main()