import os
import torch
import torch.nn.functional as F
import numpy as np
from models.network import FoodDesertClassifier
import pickle
import random
from tqdm import trange

"""
Program that trains and runs the the food desert classifier network.
"""
def main(argv):

    # TODO: read in data
    data = None
    model = FoodDesertClassifier()

    data_size = len(data)
    data = random.shuffle(data)
    train = data[:(data_size // 10) * 9]
    test = data[(data_size // 10) * 9:]
    


def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()

    pass

def eval_model(model, val_iter):
    pass

if __name__ == "__main__":
   main(sys.argv[1:])
