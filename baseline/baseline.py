import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from scipy.optimize import curve_fit
from census_reader import *
from tqdm import trange
import pickle

ZIPCODE = 95131
ZIPCODES = [95131, 36003, 60649, 14075, 19149]
LABELS = [0, 1, 1, 0, 0] # 0 = Not food desert, 1 = food desert

YEARS = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
ANY_COLLEGE_EDUC = [EDUC_ASSOCIATES_M, EDUC_ASSOCIATES_F, EDUC_BACHELORS_M,
        EDUC_BACHELORS_F, EDUC_MASTERS_M, EDUC_MASTERS_F, EDUC_PROFESSIONAL_M,
        EDUC_PROFESSIONAL_F, EDUC_DOCTORATE_M, EDUC_DOCTORATE_F]
DATA = []

FULL_DATA_PICKLE = (
        '/Users/nataliecygan/Desktop/Stanford/cs221/project/data/pickle/data.pickle')

class LinearRegressionModel(nn.Module):
        def __init__(self):
                super().__init__()
                # Define any parameters are part of the model
                initial1 = torch.zeros(3)
                initial2 = torch.zeros(1)
                # "weight" of linear model
                self.theta1 = nn.Parameter(initial1)
                # "bias" of linear model
                self.theta0 = nn.Parameter(initial2)

        # this what happens when you apply the function
        # to input. In this case I have implemented
        # y = sigma(self.theta1 * x + self.theta0)
        def forward(self, x):
            pred = self.theta1.dot(x).reshape(1)
            prediction = pred + self.theta0
            if (prediction > 0): return torch.ones(1)
            else: return torch.zeros(1)

def optimize(model, data):

        # This binds the model to the optimizer
        # Notice we set a learning rate (lr)! this is really important
        # in machine learning -- try a few different ones and see what
        # happens to learning.
        optimizer = Adam(model.parameters(), lr=0.005)

        while True:
                optimizer.zero_grad()

                loss = 0
                for i in trange(len(data)):
                        x, y = data[i]
                        x = valueToTensor(x)
                        y = valueToTensor(y)
                        y = y.reshape(1)
                        print('Y is:', y.size(), y)
                        pred = model(x)
                        loss_i = F.binary_cross_entropy(pred, y)
                        loss += loss_i

                # normalize the loss by number of data points
                loss /= len(data)

                # this step computes all gradients with "autograd"
                loss.backward()

                # this actually changes the parameters
                optimizer.step()

                print(loss.item(),i, model.theta1.item(), model.theta0.item())


def valueToTensor(v):
    return torch.tensor(v).float()

def linear(x, slope, y_intercept):
    return slope * x + y_intercept

def estimateLinearFunc(years, data, function):
    """
    Uses least squares to fit function to the data. Will return the slope and
    y-intercept that approximates the linear function.
    """
    popt, pcov = curve_fit(function, years, data)
    return popt[0], popt[1] # slope, y_intercept

def readData(reader, start_year, end_year):
    data = []
    for i in trange(len(ZIPCODES)):
        zipcode = ZIPCODES[i]
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
        out = np.reshape(out, (3))
        data.append((out, LABELS[i]))

    return data


def read_data_from_pickle():
    data = None # {zip : ([features], label)}
    with open(FULL_DATA_PICKLE, 'rb') as fp:
        data = pickle.load(fp)
    return data

def main():
    reader = CensusReader()
    #data = readData(reader, 2015, 2015)

    # ------ Processing new data format
    data = read_data_from_pickle()
    data = list(data.values())
    # ------

    model = LinearRegressionModel()
    optimize(model, data)

    print("\nSuccess")

if __name__== "__main__":
  main()

