import math
import os
import numpy as np
from tqdm import trange
from tqdm import tqdm
import pickle
import random

path_to_script = os.path.dirname(os.path.abspath(__file__))
FULL_DATA_PICKLE = os.path.join(path_to_script, "full_data_v2.pickle")
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
    
"""
Given a the food desert dataset, this will oversample it to include more food
desert datapoints, making it more balanced.
"""
def oversample(data_and_labels):
    random.shuffle(data_and_labels)
    new_data = []
    labels = [item[1] for item in data_and_labels]
    # Can tell us how many food deserts there are. For balance, we can multiply
    # the food desert datapoints by four.
    # need = np.sum([1 if label == 0 else 0 for label in labels])

    for item in data_and_labels:
        label = item[1]
        vals = item[0]
        if label == 1:
            new_data.append(item)
            new_data.append(item)
            new_data.append(item)
            new_data.append(item)
        elif label == 0:
            new_data.append(item)

    return new_data

"""
Given a the food desert dataset, this will undersample it to include fewer 
non-food desert datapoints, making it more balanced.
"""
def undersample(data_and_labels):
    random.shuffle(data_and_labels)
    new_data = []
    not_count = 0
    yes_count = 0
    for item in data_and_labels:
        label = item[1]
        vals = item[0]
        if label == 0 and not_count < 2000:
            new_data.append(item)
            not_count += 1
        elif label == 1 and yes_count < 2000:
            yes_count += 1
            new_data.append(item)

    return new_data

"""
Read in the full dataset, which is saved to a .pickle file in the format of a
dict that maps zipcodes to tuples of (feature vector, label).
This function will take off the zipcode field for training, which is not needed
in the neural network, thus just returning a list of the (feature vector, label)
tuples.
"""
def read_data():
    data_dict = None
    with open(FULL_DATA_PICKLE, 'rb') as fp:
        data_dict = pickle.load(fp)

    data = [] # List to store the (features, label) tuples.
    zipcodes = list(data_dict.keys())
    for z in zipcodes:
        # Just keep the tuple, zipcode is not needed for training.
        datapoint = data_dict[z]
        data.append(datapoint)

    print('Read in', len(data), 'datapoints.')

    return data


def main():
    data = coalese_data_files()
    print('Successfully read in a dataset of', len(data), 'datapoints.')

    # Save full data to pickle.
    with open(FULL_DATA_PICKLE, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Successfully merged all paritions of zip code data.")

if __name__== "__main__":
  main()
