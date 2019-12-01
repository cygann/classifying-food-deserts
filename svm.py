import os
import random
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


path_to_script = os.path.dirname(os.path.abspath(__file__))
# Path to the complete dataset.
FULL_DATA_PICKLE = os.path.join(path_to_script, "data/full_data_v2.pickle")

# feature_names = ['Population', 'Median Gross Rent (Dollars)', 'Median Home Value (Dollars)',
# 					 'Unemployed', 'Geographic mobility', 'No Health insurance coverage',
# 					 'Income below poverty level', 'Travel time to work', 'Median Income', 'Education']

feature_names = ['Population', 'Median Gross Rent (Dollars)', 'Median Home Value (Dollars)',
					 'Unemployed', 'Geographic mobility', 'No Health insurance coverage',
					 'Income below poverty level', 'Travel time to work', 'Median Income', 'Education',
					 '% Change Population', '% Change Median Gross Rent (Dollars)', '% Change Median Home Value (Dollars)', '% Change Unemployed', '% Change Geographic mobility', '% Change No Health insurance coverage', '% Change Income below poverty level', '% Change Travel time to work', '% Change Median Income', '% Change Education'] 

class_names = [0, 1]

"""
Program that trains the food desert classifier SVM.
"""
def main():
	random.seed(21) # So we have same parition every time.

	# Read in data from .pickle as a list of (features, label) tuples
	# each representing a zipcode datapoint.
	data = read_data()

	# Filter the data
	data = filter_out_bad_data(data)

	# Separate 80/20 as train/val/test partition.
	data_size = len(data)
	random.shuffle(data)
	train_data = data[:(data_size // 10) * 8]
	test_data = data[(data_size // 10) * 8 :]
	print(len(train_data), 'training points after filtering.')
	print(len(test_data), 'testing points after filtering.')

	# Fit the SVM
	# class_weight_dict = {0:1, 1:4.2} # food desert class is weighed ten times heavier than food desert class
	svclassifier = SVC(kernel='linear', gamma='scale', class_weight='balanced', C=10.0)
	optimize(svclassifier, train_data, test_data) # train on train data

	print('Success')


"""
Optimize on the training set. Trains on train_data and validates on val_data.
"""
def optimize(model, train_data, test_data):

	print('*******Training*******')

	# Prepare the data
	X_train = [x[0] for x in train_data]
	y_train = [x[1] for x in train_data]
	X_test = [x[0] for x in test_data]
	y_test = [x[1] for x in test_data]

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_test = np.array(X_test)
	y_test = np.array(y_test)

	# Standardize features
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
	print('******Accuracy score*******')
	print(accuracy_score(y_test,y_pred))
	print()

	# Plotting
	num_top_to_plot = int(len(X_train[0]) / 2)
	plot_coefficients(model, feature_names, num_top_to_plot)

	# Plot non-normalized confusion matrix
	# plot_confusion_matrix(y_test, y_pred, classes=class_names,
	#                       title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
	                      title='Normalized confusion matrix')

	plt.show()


"""
Filter out feature vectors that contain NaNs or infinities.
"""
def filter_out_bad_data(data):
	new_data = []
	num_removed = 0
	for (x, y) in data:
		features = []
		contains_bad_val = False
		for f in x:
			if np.isnan(f) or np.isinf(f):
				contains_bad_val = True
				break
			features.append(f)
		if contains_bad_val:
			num_removed += 1
			continue
		new_data.append((features, y))
	print("Filtered out", num_removed, "data points.")
	return new_data

"""
Plot the weights assigned to the features (coefficients in the primal problem).
This is only available in the case of a linear kernel.
"""
def plot_coefficients(classifier, feature_names, top_features):
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
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
	if not title:
	    if normalize:
	        title = 'Normalized confusion matrix'
	    else:
	        title = 'Confusion matrix, without normalization'

	np.set_printoptions(precision=4)
	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	# classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
	    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	    print("Normalized confusion matrix")
	else:
	    print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
	       yticks=np.arange(cm.shape[0]),
	       # ... and label them with the respective list entries
	       xticklabels=classes, yticklabels=classes,
	       title=title,
	       ylabel='True label',
	       xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.4f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
	    for j in range(cm.shape[1]):
	        ax.text(j, i, format(cm[i, j], fmt),
	                ha="center", va="center",
	                color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax




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
