import math
import os
import numpy as np
import torch
import torch.nn as nn
from census_reader import *
from tqdm import trange
from tqdm import tqdm
import pickle
import random

def coalese_data_files(paths):
    merged_data = {}

    for p in paths:
        data = None
        with open(p, 'rb') as fp:
            data = pickle.load(fp)

        zipcodes = list(data.keys())
        for z in zipcodes:
            merged_data[z] = data[z]

    return merged_data
    

def main():
    paths = []
    data = coalese_data_files(paths)

    # Save full data to pickle.
    with open(FULL_DATA_PATH, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Successfully merged all paritions of zip code data.")
