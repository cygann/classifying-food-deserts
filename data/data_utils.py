import math
import os
import numpy as np
from tqdm import trange
from tqdm import tqdm
import pickle
import random

path_to_script = os.path.dirname(os.path.abspath(__file__))
FULL_DATA_PICKLE = os.path.join(path_to_script, "full_data.pickle")
data_files = [os.path.join(path_to_script, "data_parts/data_i.pickle"),
        os.path.join(path_to_script, "data_parts/data_ii.pickle"),
        os.path.join(path_to_script, "data_parts/data_iii.pickle")]

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
    

def main():
    data = coalese_data_files()
    print('Successfully read in a dataset of', len(data), 'datapoints.')

    # Save full data to pickle.
    with open(FULL_DATA_PICKLE, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Successfully merged all paritions of zip code data.")

if __name__== "__main__":
  main()
