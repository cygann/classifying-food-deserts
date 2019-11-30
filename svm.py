import os
import random
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


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
	print(len(train_data), 'training points.')
	print(len(test_data), 'testing points.')




	# Fit the SVM
	svclassifier = SVC(kernel='linear', gamma='scale', class_weight='balanced')
	optimize(svclassifier, train_data, test_data) # train on train data

	print('Success')


"""
Optimize on the training set. Trains on train_data and validates on val_data.
"""
def optimize(model, train_data, test_data):

	print('*******Training*******')

	# Prepare the data
	X_train, y_train = [], []
	for (x, y) in train_data:
		features = [np.nan_to_num(f) for f in x]
		X_train.append(features)
		y_train.append(y)

	X_test, y_test = [], []
	for (x, y) in test_data:
		features = [np.nan_to_num(f) for f in x]
		X_test.append(features)
		y_test.append(y)

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_test = np.array(X_test)
	y_test = np.array(y_test)

	# Standarize features
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	# Fix the NaNs and infinities again
	for i in range(len(X_train)):
		X_train[i] = [np.nan_to_num(f) for f in X_train[i]]


	# Fit the SVM model
	clf = model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	# Print stats
	print('******Confusion matrix*******')
	print(confusion_matrix(y_test,y_pred))
	print('******Classification report*******')
	print(classification_report(y_test,y_pred))

	# Plotting
	plot_coefficients(model, feature_names)



"""
Plot the weights assigned to the features (coefficients in the primal problem).
This is only available in the case of a linear kernel.
"""
def plot_coefficients(classifier, feature_names, top_features=5):
	coef = classifier.coef_.ravel()
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

	# create plot
	plt.figure(figsize=(15, 5))
	colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
	plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	feature_names = np.array(feature_names)
	plt.xticks(np.arange(0, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
	plt.show()



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


if __name__ == "__main__":
	main()
