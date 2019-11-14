import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from scipy.optimize import curve_fit
from census_reader import *
from tqdm import trange
import pickle


ZIPCODE = 95131
ZIPCODES = [95131, 36003, 60649, 14075, 19149] # Must be of length 2 or more
LABELS = [0, 1, 1, 0, 0] # 0 = Not food desert, 1 = food desert

YEARS = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
ANY_COLLEGE_EDUC = [EDUC_ASSOCIATES_M, EDUC_ASSOCIATES_F, EDUC_BACHELORS_M,
        EDUC_BACHELORS_F, EDUC_MASTERS_M, EDUC_MASTERS_F, EDUC_PROFESSIONAL_M,
        EDUC_PROFESSIONAL_F, EDUC_DOCTORATE_M, EDUC_DOCTORATE_F]
DATA = []

PATH_DATA_PICKLE = '/Users/Priscilla/Desktop/Classes/Fall 2019/CS221/cs221-project/baseline_pickle_files/census_data_for_baseline.pickle'

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

def optimize(model, data):
    """
    Optimize on the training set
    """
    print('*******Training*******')
    criterion = torch.nn.BCELoss(size_average=True) # binary logarithmic loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 5
    iter = 0
    for epoch in range(epochs):
        print('EPOCH', epoch)
        for x,y in data:
            # print('Zipcode:', ZIPCODES[i])
            # x, y = data[i]
            # x = valueToTensor(x)
            # y = valueToTensor(y)
            # y = y.reshape(1)
            x = Variable(valueToTensor(x))
            y = Variable(valueToTensor(y))
            # print('x is:', x)
            # print('y is:', y)

            optimizer.zero_grad()

            # Forward pass
            pred = model(x) 
            # print('pred is', pred.data[0])

            # Compute loss
            loss = criterion(pred, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            iter+=1
            if iter % 1 == 0:
                # Calculate accuracy on TRAINING set
                num_correct = 0
                total = 0
                for x, y in data:
                    x = Variable(valueToTensor(x))
                    pred = model(x)
                    _, predicted = torch.max(pred.data, 0)
                    total += 1
                    # for gpu, bring the predicted and labels back to cpu fro python operations to work
                    num_correct = (num_correct+1 if (predicted == y) else num_correct)
                    # print('num_correct:', num_correct, ' total', total)
                accuracy = 100.0 * num_correct/total
                print('Iteration: {}. Loss: {}. Train Accuracy: {}.'.format(iter, loss.item(), accuracy))
        print()
def valueToTensor(v):
    return torch.tensor(v).float()

def test(model, test_data):
    """
    Compute the model's accuracy on the test set
    """
    print('*******Testing*******')
    num_correct = 0
    total = 0
    for x,y in test_data:
        x = Variable(valueToTensor(x))
        pred = model(x)
        _, predicted = torch.max(pred.data, 0)
        total += 1
        num_correct = (num_correct+1 if (predicted == y) else num_correct)
        # print('num_correct:', num_correct, ' total', total)
    accuracy = 100.0 * num_correct/total
    print('Test Accuracy: {}.'.format(accuracy))

# def linear(x, slope, y_intercept):
#     return slope * x + y_intercept

# def estimateLinearFunc(years, data, function):
#     """
#     Uses least squares to fit function to the data. Will return the slope and
#     y-intercept that approximates the linear function.
#     """
#     popt, pcov = curve_fit(function, years, data)
#     return popt[0], popt[1] # slope, y_intercept

def readData(reader, start_year, end_year):
    data = []
    for i in trange(len(ZIPCODES)):
        zipcode = ZIPCODES[i]
        print(zipcode)
        # Get population data
        population = reader.getDataOverInterval([TOTAL_POP], zipcode,
                start_year, end_year)
        # Get higher education data
        high_educ_data = reader.getDataOverInterval(ANY_COLLEGE_EDUC, zipcode,
                start_year, end_year)
        educ_percent = reader.normalizeToPopulation(population, high_educ_data)
        # Get income data
        income = reader.getDataOverInterval([MEDIAN_INCOME], zipcode,
                start_year, end_year)
        out = np.concatenate((population, educ_percent, income))
        # print("out", out)
        # out = np.reshape(out, 3) # size 3 array
        # print("after reshape", out)
        data.append((out, LABELS[i]))

    return data



def main():
    reader = CensusReader() 

    # # Save training/test data as pickle files - only do once
    # data = readData(reader, 2015, 2015) # only reading for 1 year right now
    # with open(PATH_DATA_PICKLE, 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Open pickle file - only do after it has been saved (above)
    data = None
    with open(PATH_DATA_PICKLE, 'rb') as fp:
        data = pickle.load(fp)

    input_dim = 3 # the number of factors
    output_dim = 1 # binary classification

    model = LogisticRegressionModel(input_dim, output_dim)
    optimize(model, data[0:-1]) # train on all but last zipcode
    test(model, [data[-1]]) # test on last zipcode

    print("\nSuccess")

if __name__== "__main__":
  main()

