import os
import random
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from models.single_layer import LogisticRegressionModel
from models.network import FoodDesertClassifier
from train import optimize_nn


path_to_script = os.path.dirname(os.path.abspath(__file__))
# Path to the complete dataset.
FULL_DATA_PICKLE = os.path.join(path_to_script, "data/full_data.pickle")

feature_names = ['Population', 'Median Gross Rent (Dollars)', 'Median Home Value (Dollars)',
					 'Unemployed', 'Geographic mobility', 'No Health insurance coverage',
					 'Income below poverty level', 'Travel time to work', 'Median Income', 'Education']


"""
Program that trains the food desert classifier SVM.
"""
def main():
    random.seed(21) # So we have same parition every time.
    
    # Read in data from .pickle as a list of (features, label) tuples
    # each representing a zipcode datapoint.
    data = read_data()
    
    # Separate 80/20 as train/val/test partition.
    data_size = len(data)
    random.shuffle(data)
    train_data = data[:(data_size // 10) * 8]
    test_data = data[(data_size // 10) * 8 :]
    
    x_train, y_train = standardize_data(train_data)
    x_test, y_test = standardize_data(test_data)
    
    #visualize_data(data_size, x_train, y_train)
    
    find_hidden_vals(train_data, test_data)


def read_data():
    data_dict = None
    with open(FULL_DATA_PICKLE, 'rb') as fp:
        data_dict = pickle.load(fp)
        
    data = []
    zipcodes = list(data_dict.keys())
    #USE SMALL SUBSET TO TEST
    #zipcodes = zipcodes[0:1000]
    for z in zipcodes:
        datapoint = data_dict[z]
        data.append(datapoint)
    
    print('Read in', len(data), 'datapoints.')
    
    return data

def visualize_data(data_length, x_data, y_data):
    #Visualize unemployment, geographic mobility and education
    unempl_index = 3
    geo_mobility_index = 4
    educ_index = 9
    
    unempl_list = []
    geo_mobility_list = []
    educ_list = []
    label_list = y_data
    
    for i in range(len(x_data)):
        unempl_list.append(x_data[i][unempl_index])
        geo_mobility_list.append(x_data[i][geo_mobility_index])
        #educ_list.append(x_data[i][educ_index])

    unempl_geo_list = []
    for unempl_vals, geo_vals in zip(unempl_list, geo_mobility_list):
        unempl_geo_list.append(np.array([unempl_vals, geo_vals]))
    unempl_geo_list = np.array(unempl_geo_list)
        
    #colors = ['red' if label_list[i] == 1 else 'blue' for i in range(len(data))]
    
    data_limit = data_length #make lower to test on smaller dataset
    perc = Perceptron(max_iter=50)
    perc.fit(unempl_geo_list[0:data_limit], label_list[0:data_limit])
    tree = DecisionTreeClassifier()
    tree.fit(unempl_geo_list[0:data_limit], label_list[0:data_limit])
    print("number of contours", tree.tree_.node_count)
    
    
    print("Plotting decision boundary")
    plot_decision_boundary(perc, unempl_geo_list[0:data_limit], label_list[0:data_limit])
    plot_decision_boundary(tree, unempl_geo_list[0:data_limit], label_list[0:data_limit])
#    fig1 = plt.figure(figsize=(15,10))
#    plt.scatter(unempl_list, geo_mobility_list, alpha = 0.4, color = colors)
#
#    fig2 = plt.figure(figsize=(15,5))
#    plt.scatter(unempl_list, educ_list, alpha = 0.4, color = colors)
#   
#    fig3 = plt.figure(figsize=(15,5))
#    plt.scatter(educ_list, geo_mobility_list, alpha = 0.4, color = colors)
#    plt.show()
    
def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    print("Printing contour map")
    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    c = plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    
    print("Printing scatter plot")
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');
    
    plt.savefig('decision-boundary-all.png')
    
    print("levels", len(c.levels))
    print("layers", len(c.layers))
    
def standardize_data(data):
    # Prepare the data
    x_data, y_data = [], []
    
    for (x, y) in data:
        if not np.isnan(x).any() and not np.isnan(y).any():
            x = np.array(x, dtype = np.float64)
            #features = [np.nan_to_num(f) for f in x]
            #x_data.append(features)
            x_data.append(x)
            y_data.append(y)

    x_data = np.array(x_data, dtype = np.float64)
    y_data = np.array(y_data)

    # Standarize features
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    
#    print("second scale")
#    # Fix the NaNs and infinities again
#    for i in range(len(x_data)):
#        x_data[i] = [np.nan_to_num(f) for f in x_data[i]]
#    
    x_data = np.array(x_data, dtype = np.float64)
        
    return (x_data, y_data)

def find_hidden_vals(train_data, test_data):
    input_dim = len(train_data[0][0]) # number of features
    output_dim = 2 # two classes: food desert and not food desert
    #hidden_dim_list = [5, 10, 8, 12]

    
    
    
    accuracy_list = []
    
#    for num_layers in range(2, 11):
#        hidden_dim_list = []
#        for layer in num_layers:
#            for num_nodes in [2, 4, 16]:
#                hidden_dim_list.append(num_nodes)
#                num_nodes = pow(num_nodes, 2)
#        model_nn = FoodDesertClassifier(input_dim, hidden_dim_list, output_dim)
#        optimize_nn(model_nn, train_data, train_data, test_data)
    accuracy_list = []
    hidden_list = [[2, 2, 2, 2, 2, 2, 2, 2], [2, 4, 4, 2, 16, 4, 2, 4], [4, 4, 16, 2, 16, 4, 4, 4], [16, 2, 4, 2, 2, 4, 4, 2]]
    for nodes in hidden_list:
        model_nn = FoodDesertClassifier(input_dim, nodes, output_dim)
        accuracy_list.append(optimize_nn(model_nn, train_data, train_data, test_data))
    
if __name__ == "__main__":
	main()