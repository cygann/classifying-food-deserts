import os
import random
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from itertools import permutations


path_to_script = os.path.dirname(os.path.abspath(__file__))
# Path to the complete dataset.
FULL_DATA_PICKLE = os.path.join(path_to_script, "data/full_data_v2.pickle")

# feature_names = ['Population', 'Median Gross Rent (Dollars)', 'Median Home Value (Dollars)',
# 					 'Unemployed', 'Geographic mobility', 'No Health insurance coverage',
# 					 'Income below poverty level', 'Travel time to work', 'Median Income', 'Education']

feature_names = ['Population', 'Median Gross Rent (Dollars)', 'Median Home Value (Dollars)',
					 'Unemployed', 'Geographic mobility', 'No Health insurance coverage',
					 'Income below poverty level', 'Travel time to work', 'Median Income', 'Education',
					 '% Change Population', '% Change Median Gross Rent (Dollars)', '% Change Median Home Value (Dollars)',
					 '% Change Unemployed', '% Change Geographic mobility', '% Change No Health insurance coverage',
					 '% Change Income below poverty level', '% Change Travel time to work', '% Change Median Income', '% Change Education']


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
	#data_augment(data)
#	y_vals = []
#	for x,y in data:
#		y_vals.append(y)
#	print("Ratio of 1s to 0s:", sum(y_vals) / len(y_vals))

	# Separate 80/20 as train/val/test partition.
	data_size = len(data)
	random.shuffle(data)
	train_data = data[:(data_size // 10) * 8]
	test_data = data[(data_size // 10) * 8 :]
	print(len(train_data), 'training points after filtering.')
	print(len(test_data), 'testing points after filtering.')

	clf = MLPClassifier(solver='adam', alpha=1e-5, 
			hidden_layer_sizes=(16,16), max_iter=500, random_state=0)
	print("Accuracy: ", optimize(clf, train_data, test_data))

#gridSearch(train_data, test_data)

	# Make the Logistic Regression model
#	clf = MLPClassifier(solver='adam', alpha=1e-5, 
#		hidden_layer_sizes=(5, 2), random_state=0)
#	optimize(clf, train_data, test_data) # train on train data

	print('Success')

def gridSearch(train_data, test_data):
	
	accuracy_list = dict()
	for num_layers in range(2, 3):
		poss_vals = []
		choose_from = []
		for i in range(num_layers):
			choose_from.append(16)
			choose_from.append(20)
			#choose_from.append(24)
			#choose_from.append(16)
		for item in permutations(choose_from, r=num_layers):
			poss_vals.append(item)
			accuracy_list[item] = []
		poss_vals = list(set(poss_vals))
		for poss_combo in poss_vals:
			clf = MLPClassifier(solver='adam', alpha=1e-5, 
				hidden_layer_sizes=poss_combo, max_iter=500, random_state=0)
			accuracy = optimize(clf, train_data, test_data)
			print("Permutation:", poss_combo)
			print("Accuracy: ", accuracy)
			accuracy_list[poss_combo].append(accuracy)
	
	#for poss_combo in poss_vals:

#		for alpha_val in [1e-6, 1e-4, 1e-2]:
#			clf = MLPClassifier(solver='adam', alpha=alpha_val, 
#				hidden_layer_sizes=poss_combo, random_state=0)
#			accuracy_list[poss_combo].append((alpha_val, optimize(clf, train_data, test_data)))
		

	print("Accuracy Dict: ", accuracy_list)
	print("Best Accuracy: ", max(accuracy_list.values()))
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


	# Fit the Logistic Regression model
	model = model.fit(X_train, y_train)
	print("Probability: ", sum(model.predict_proba(X_test)))
	y_pred = []
	for val in model.predict_proba(X_test):

		if val[1] > 0.8:
			y_pred.append(1)
		else:
			y_pred.append(0)
	#y_pred = model.predict(X_test)

	# Print stats
	print('******Confusion matrix*******')
	print(confusion_matrix(y_test,y_pred))
	print('******Classification report*******')
	print(classification_report(y_test,y_pred))
	print('******Accuracy score*******')
	print(accuracy_score(y_test,y_pred))
	print()

#	# Plotting
#
	plt.xlabel("Training Iteration")
	plt.ylabel("Log Loss")
	plt.plot(model.loss_curve_)
	num_top_to_plot = 3
	plot_coefficients(model, feature_names, num_top_to_plot)
#
#	# Plot non-normalized confusion matrix
#	# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#	#                       title='Confusion matrix, without normalization')
#
#	# Plot normalized confusion matrix
	plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
						title='Normalized confusion matrix')
#
	plt.show()

	return (accuracy_score(y_test,y_pred))


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


def data_augment(data):
	class_1 = []
	for x,y in data:
		if y == 1:
			class_1.append((x,y))
	data.extend(class_1)
	data.extend(class_1)
	data.extend(class_1)
	data.extend(class_1)

	y_vals = []
	for x,y in data:
		y_vals.append(y)
	print("Ratio of 1s to 0s:", sum(y_vals) / len(y_vals))

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
