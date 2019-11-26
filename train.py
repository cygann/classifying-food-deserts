import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from models.network import FoodDesertClassifier
import pickle
import random
from tqdm import trange

path_to_script = os.path.dirname(os.path.abspath(__file__))
# Path to the complete dataset.
FULL_DATA_PICKLE = os.path.join(path_to_script, "data/full_data.pickle")

"""
Program that trains and runs the the food desert classifier network.
"""
def main(argv):
    random.seed(21) # So we have same parition every time.

    # Read in data from .pickle as a list of (features, label) tuples
    # representing a zipcode datapoint.
    data = read_data()

    # Separate 90/10 as train/test partition.
    data_size = len(data)
    random.shuffle(data)
    train = data[:(data_size // 10) * 9]
    test = data[(data_size // 10) * 9:]
    print(len(train), 'training points.')
    print(len(test), 'testing points.')

    model = FoodDesertClassifier()
    
"""
Read in the full dataset, which is saved to a .pickle file in the format of a
dict that maps zipcodes to tuples of (feature vector, label).
This function will take of the zipcode field for training, which is not needed
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

def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()

def eval_model(model, val_iter):
    pass

if __name__ == "__main__":
   main(sys.argv[1:])
