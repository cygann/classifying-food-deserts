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
from sklearn import metrics
from tqdm import trange
import matplotlib.pyplot as plt

path_to_script = os.path.dirname(os.path.abspath(__file__))
# Path to the complete dataset.
FULL_DATA_PICKLE = os.path.join(path_to_script, "data/full_data_v2.pickle")

"""
Program that trains and runs the the food desert classifier network.
"""
def main(argv):
    # random.seed(21) # So we have same parition every time.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read in data from .pickle as a list of (features, label) tuples
    # each representing a zipcode datapoint.
    data_and_labels = data_utils.read_data()

    need = np.sum([1 if label[1] == 0 else 0 for label in data_and_labels])
    print('need', need)

    # Oversample
    # data_and_labels = data_utils.oversample(data_and_labels)
    # data_and_labels = data_utils.undersample(data_and_labels)

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

    train_data = data_utils.oversample_train(train_data)

    print(len(train_data), 'training points.')
    print(len(val_data), 'validation points.')
    print(len(test_data), 'testing points.')

    input_dim = len(data[0][0]) # number of features
    output_dim = 2 # two classes: food desert and not food desert
    hidden_dim_list = [16, 36, 36, 24]

    model_nn = FoodDesertClassifier(input_dim, hidden_dim_list,
            output_dim).to(device)
    loss = optimize_nn(model_nn, train_data, val_data, test_data)
    
    eval_model_nn(model_nn, loss, test_data, "Testing")

def valueToTensor(v):
    return torch.tensor(v).float()

"""
Optimize neural network model on the training set
"""
def optimize_nn(model, train_data, val_data, test_data):
    print('*******Training*******')
    
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([4.187, 1]))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
   
    num_epochs = 10
    iter = 0
    loss_list = []
    accuracy_list = []
    iters = []

    val_loss_list = []
    val_accuracy_list = []

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
                # loss = criterion(pred.float(), y)
    
                # Backward pass to get gradients w.r.t parameters
                loss.backward()
    
                # Updating parameters
                optimizer.step()
                
                running_loss += loss.item()
                
                iter += 1
                # loss_list.append(loss.item())
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

                    accuracy_list.append(accuracy)
                    loss_list.append(loss.item())
                    iters.append(iter)
                    if iter % 5000 == 0: 
                        val_accuracy = eval_model_nn(model, loss, val_data,
                                "Validation")
                        val_accuracy_list.append(val_accuracy)
                        val_loss_list.append(loss)
                        plot_data(accuracy_list, loss_list, val_accuracy_list,
                                val_loss_list, iters, iter)
                    
    eval_model_nn(model, loss, test_data, "Test")
    return loss

def plot_data(accuracy_list, loss_list, val_accuracy_list, val_loss_list,
        iters, iter):
    plt.subplot(211)
    plt.plot(iters, accuracy_list)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    plt.subplot(212)
    plt.plot(iters, loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # plt.subplot(313)
    # plt.plot(len(val_accuracy_list), val_accuracy_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Validation accuracy')

    filename = './logs/plot' + str(iter) + '.png'

    plt.savefig(filename)
    plt.close('all')

def eval_model_nn(model, loss, data, testType):
    num_correct = 0
    total = 0
    y_test = []
    y_pred = []
    for x, y in data:
        if not np.isnan(x).any():
            total += 1
            x = Variable(torch.tensor(x).float())
            prediction = model.predict(x)

            num_correct = (num_correct + 1 if (prediction== y) 
                    else num_correct)

            y_test.append(y)
            y_pred.append(prediction)

    accuracy = 100.0 * num_correct / total
    string = 'Loss: {}. ' + testType + ' Accuracy: {}.'
    print(string.format(loss.item(), accuracy))
    print()
    print('******Confusion matrix*******')
    print(metrics.confusion_matrix(y_test, y_pred))
    print('******Classification report*******')
    print(metrics.classification_report(y_test, y_pred))
    print('******Accuracy score*******')
    print(metrics.accuracy_score(y_test, y_pred))
    print()

    return accuracy

if __name__ == "__main__":
   main(sys.argv[1:])
