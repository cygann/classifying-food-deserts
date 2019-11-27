import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from models.single_layer import LogisticRegressionModel
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
    # each representing a zipcode datapoint.
    data = read_data()

    # Separate 80/10/10 as train/val/test partition.
    data_size = len(data)
    random.shuffle(data)
    train_data = data[:(data_size // 10) * 8]
    val_data = data[(data_size // 10) * 8 : (data_size // 10) * 9]
    test_data = data[(data_size // 10) * 9 :]
    print(len(train_data), 'training points.')
    print(len(val_data), 'validation points.')
    print(len(test_data), 'testing points.')


    input_dim = len(data[0][0]) # number of features
    output_dim = 2 # two classes: food desert and not food desert

    model = LogisticRegressionModel(input_dim, output_dim)
    optimize(model, train_data, val_data) # train on train data
    test(model, test_data) # test on test data
 

"""
Optimize on the training set. Trains on train_data and validates on val_data.
"""
def optimize(model, train_data, val_data):
    """
    Optimize on the training set
    """
    print('*******Training*******')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 5
    iter = 0
    for epoch in range(num_epochs):
        print('EPOCH', epoch)
        for x,y in train_data:
            # print('Zipcode:', ZIPCODES[i])
   
            # Load data as Variable
            x = Variable(valueToTensor(x))
            y = Variable(valueToTensor(y))
            # print('x is:', x)
            # print('y is:', y)

            # Clear gradients w.r.t parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            pred = model(x) 
            print('pred is', pred.data)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(pred, y)

            # Backward pass to get gradients w.r.t parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1
            if iter % 100 == 0:
                # Calculate accuracy on val_data
                num_correct = 0
                for x, y in val_data:
                    x = Variable(valueToTensor(x))
                    pred = model(x)
                    pred.unsqueeze_(0) # add a dimension before passing to criterion
                    _, predicted = torch.max(pred.data, 0)
                    num_correct = (num_correct+1 if (predicted == y) else num_correct)
                    print('num_correct:', num_correct, ', total', total)
                accuracy = 100.0 * num_correct / len(val_data)
                print('Iteration: {}. Loss: {}. Train Accuracy: {}.'.format(iter, loss.item(), accuracy))
        print()


def valueToTensor(v):
    return torch.tensor(v).float()


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

def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()

def eval_model(model, val_iter):
    pass

if __name__ == "__main__":
   main(sys.argv[1:])
