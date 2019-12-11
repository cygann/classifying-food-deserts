import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from models.network import FoodDesertClassifier
import data.data_utils as data_utils
import pickle
import random
from sklearn import preprocessing
from tqdm import trange

path_to_script = os.path.dirname(os.path.abspath(__file__))
# Path to the complete dataset.
FULL_DATA_PICKLE = os.path.join(path_to_script, "data/full_data_v2.pickle")

"""
Program that trains and runs the the food desert classifier network.
"""
def main(argv):
    # random.seed(21) # So we have same parition every time.

    # Read in data from .pickle as a list of (features, label) tuples
    # each representing a zipcode datapoint.
    data_and_labels = data_utils.read_data()

    # Oversample
    # data_and_labels = data_utils.oversample(data_and_labels)
    data_and_labels = data_utils.undersample(data_and_labels)

    # Standardize the data.
    x_data = [x[0] for x in data_and_labels]
    y_data = [x[1] for x in data_and_labels]
    scaler = preprocessing.StandardScaler()
    x_data = scaler.fit_transform(x_data)

    # New dataset that is standardized.
    data = [(x_data[i], y_data[i]) for i in range(len(x_data))]

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
    hidden_dim_list = [5, 10, 8, 12]

    model_nn = FoodDesertClassifier(input_dim, hidden_dim_list, output_dim)
    loss = optimize_nn(model_nn, train_data, val_data, test_data)
    
    eval_model_nn(model_nn, loss, test_data, "Testing")

def valueToTensor(v):
    return torch.tensor(v).float()

"""
Optimize neural network model on the training set
"""
def optimize_nn(model, train_data, val_data, test_data):
    print('*******Training*******')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    num_epochs = 5
    iter = 0
    for epoch in range(num_epochs):
        running_loss = 0
        print('EPOCH', epoch)
        for x, y in train_data:
            if not np.isnan(x).any():
                
                # Load data as Variable
                x = Variable(valueToTensor(np.asmatrix(x)))
                y = Variable(valueToTensor(y))
    
                # Clear gradients w.r.t parameters
                optimizer.zero_grad()
    
                # Forward pass to get output/logits
                pred = model.forward(x)
    
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(pred.float(), y.long().unsqueeze(0))
    
                # Backward pass to get gradients w.r.t parameters
                loss.backward()
    
                # Updating parameters
                optimizer.step()
                
                running_loss += loss.item()
                
                iter += 1
                if iter % 100 == 0:
                    # Calculate accuracy on train_data
                    num_correct = 0
                    total = 0
                    for x, y in train_data:
                        if not np.isnan(x).any():
                            total += 1
                            x = Variable(valueToTensor(x))
                            prediction = model.predict(x)

                            num_correct = (num_correct + 1 if (prediction == y) 
                                    else num_correct)

                    accuracy = 100.0 * num_correct / total
                    print('Iteration: {}. Loss: {}. Training Accuracy: {}.'
                            .format(iter, loss.item(), accuracy))
                    
    eval_model_nn(model, loss, val_data, "Validation")
    return loss

def eval_model_nn(model, loss, data, testType):
    num_correct = 0
    total = 0
    for x, y in data:
        if not np.isnan(x).any():
            total += 1
            x = Variable(torch.tensor(x).float())
            prediction = model.predict(x)

            num_correct = (num_correct + 1 if (prediction== y) 
                    else num_correct)

    accuracy = 100.0 * num_correct / total
    string = 'Loss: {}. ' + testType + ' Accuracy: {}.'
    print(string.format(loss.item(), accuracy))
    return accuracy

if __name__ == "__main__":
   main(sys.argv[1:])
