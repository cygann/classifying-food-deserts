import math
import os
import numpy as np
from census_reader import *
from tqdm import trange
from tqdm import tqdm
import pickle
import random
import multiprocessing
from functools import partial

BATCH_SIZE = 20

ZIPCODES_ = [95131, 36003, 60649, 14075, 19149] # Must be of length 2 or more
LABELS_ = [0, 1, 1, 0, 0] # 0 = Not food desert, 1 = food desert
labels_sample = {}
for i in range(len(ZIPCODES_)):
    zipcode = str(ZIPCODES_[i])
    label = LABELS_[i]
    labels_sample[zipcode] = label

# Contains map of zipcodes to binary food desert labels.
path_to_script = os.path.dirname(os.path.abspath(__file__))
LABELS_PICKLE = os.path.join(path_to_script, "data/labels.pickle")
FULL_DATA_PICKLE = os.path.join(path_to_script, "data/data.pickle")

"""
Parses a census api variable file. Each row contains one or more census
variables. If a row contains multiple variables, then these quantities are
summed together to create one census feature, which is useful for things like #
people with associates, # people with masters, etc. 

The special token at the beginning of a line 'N' indicates that this feature
should be normalized to the population, to create a percentage feature. An
example of this is a row that represents the number of individuals with higher
education in a zipcode, which is converted into a percentage instead of a raw
count.

Any token that starts with a '#' triggers the reader to stop reading the current
line as it denotes a comment.

Note: Every token must be separated by ',', even special ones.

Returns: a list of tuples called unique_ids which contains (variables, nomalize) 
tuples, where variables is the list of census variables to sum for one feature
and normalize is a boolean indicating whether this feature should be normalized
to the zip code's population.
"""
def read_census_field_file(path):
    unique_ids = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            variables = [] # Hold all variables that need to be summed.
            # Denote whether or not this field should be normalized to the
            # population count.
            normalize = False

            tokens = line.split(',')
            start = 0

            # 'N' is a  special token that denotes this field needing to be
            # normalized by the population.
            if tokens[0].strip() == 'N':
                normalize = True
                start = 1 # Skip normalized token

            for t in tokens[start:]:
                cleaned = t.strip()
                # Line starting with '#' indicates comment, stop reading line.
                if cleaned[0] == '#': break
                variables.append(cleaned)

            unique_ids.append((variables, normalize))

    return unique_ids

"""
Function for separate tqdm process that updates the progress bar during a
parallelized data batch generation. Each time a zipcode is done being processed,
this is called upon for the bar to update.
"""
def tqdm_listener(q):
    pbar = tqdm(total=BATCH_SIZE)
    for item in iter(q.get, None):
        pbar.update()

"""
Function for paralell data processing processes. This will fetch the data
features for a single datapoint and place the (features, label) tuple into the
results dict.
"""
def fetch_features(reader, start_year, end_year, unique_ids, results, 
        datapoint):
    zipcode = datapoint[0]
    label = datapoint[1]
    features = []

    # Store zipcodes that don't properly read.
    valid_zip = True

    # Must get populatation data in order for reader.normalizeToPopulation
    # to work. This will always be the first item in unique_ids.
    assert(len(unique_ids) >= 1)
    population = reader.getDataOverInterval(unique_ids[0][0], zipcode,
            start_year, end_year)
    features.append(population)

    # Each entry in unique_ids is a list of census variables that get summed
    # together. The reader.getDataOverInterval function handles this summing
    # for us; it just requires a list of variables.
    for variable_set in unique_ids[1:]:
        variables = variable_set[0]
        normalize = variable_set[1]

        # Ensure that the zipcode is successfully read.
        try:
            var_data = reader.getDataOverInterval(variables, zipcode,
                    start_year, end_year)
        except Exception:
            valid_zip = False
            break

        # It's possible no exception was thrown, but the zipcode could just
        # have no data.
        if var_data == -1:
            valid_zip = False
            break

        # Only normalize this feature if requested.
        if normalize:
            var_data = reader.normalizeToPopulation(population, var_data)

        features.append(var_data)

    # If this is no longer a valid zipcode due to census API errors, skip
    # the rest of this code.
    if not valid_zip: 
        results[zipcode] = None
    else:
        out = np.concatenate(features)
        results[zipcode] = (out, label)

    # q.put(1) # Update progress bar.


"""
Obtains the census features for all datapoints using the census API. This
funciton returns the full dataset as a dict, which maps zipcodes to a tuple
containing their census feature vector and food desert label. This data dict is
saved to a .pickle file and is also returned by the function.

Parameters:
    reader : an Census reader object that will obtain values from the census API.
    labels : a dict that maps zipcodes to binary food desert flags (labels)
    start_year : the starting year for the interval over which the data is
        desired to be extracted.
    end_year : the ending year for the interval over which the data is
        desired to be extracted. (Inclusive)
    unique_ids : a list containing the census data variables. An example of one
        such value is 'B00001_001E', which is the Census Data API variable for the
        total population.
"""
def obtain_features_from_census(reader, labels, start_year, end_year,
        unique_ids):
    zipcodes = list(labels.keys()) # The keys to the labels dict are the zips

    # Create an iterable of tuples ready for multithreading
    iterable = []
    for i in range(len(zipcodes)):
        zipcode = zipcodes[i]
        label = labels[zipcode]
        iterable.append((zipcode, label))

    manager = multiprocessing.Manager()
    results = manager.dict()

    pool = multiprocessing.Pool(8)
    pbar = tqdm(total=BATCH_SIZE)
    def update(*a):
        pbar.update()

    for i in range(BATCH_SIZE):
        pair = iterable[i]
        pool.apply_async(fetch_features, args=(reader, start_year, end_year,
                unique_ids, results, pair,), callback=update)
    pool.close()
    pool.join()

    # Final post-processing to get data map.
    data = {}
    error_zips = 0
    result_keys = results.keys()
    for i in trange(len(result_keys)):
        zipcode = result_keys[i]
        item = results[zipcode]
        if item == None: 
            error_zips += 1
            continue # Skip the invalid zips
        features = item[0]
        label = item[1]
        data[zipcode] = (features, label)

    print('Successfully created', len(data), 'data points.')
    print('Had an issue reading', error_zips, 'data points. These were excluded from the dataset.')

    # Save as .pickle file.
    with open(FULL_DATA_PICKLE, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data

"""
Reads in the labels dict from the .pickle file specified by LABELS_PICKLE. This
dict maps zipcodes to their binary food desert label. This is returned by the
function.
Ex. data[zipcode] = 0 --> zipcode is not a food desert
    data[zipcode] = 1 --> zipcode is a food desert
"""
def read_labels():
    labels = None
    with open(LABELS_PICKLE, 'rb') as fp:
        labels = pickle.load(fp)
    print('Successfully read in zipcode labels')

    return labels

def main():
    # Census reader object for reading from API.
    reader = CensusReader() 

    # Read the census variables from file.
    unique_ids = read_census_field_file('census_api_variables.txt')
    print(unique_ids)
    
    # Get labels to build full dataset from
    labels = read_labels()
    #labels = labels_sample
    # labels = {z : labels[z] for z in list(random.sample(labels.keys(), 300))}
    labels = {z : labels[z] for z in list(labels.keys())[:BATCH_SIZE]}

    # Get the full data fold from the census.
    data = obtain_features_from_census(reader, labels, 2015, 2015, unique_ids)

    num_deserts = 0
    for z in list(data.keys()):
        if data[z][1] == 1: num_deserts += 1
    
    print('Found', num_deserts, 'food deserts')

    print("\nSuccess")

if __name__== "__main__":
  main()

