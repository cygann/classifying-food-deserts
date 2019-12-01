from __future__ import division
import numpy as np
import math
from census import Census
from us import states
from tqdm import trange, tqdm
import pandas as pd
import pickle

# Threshold for a zipcode to be considered a food desert
FOOD_DESERT_THRES = .25

""" Data files """
# File for the US food atlas data, which documents food access for US Census
# Tracts.
FOOD_ATLAS_FILE = (
        '/Users/nataliecygan/Desktop/Stanford/cs221/project/data/food_atlas.xlsx')
# Contains the Census Tract --> US ZIP conversions; atlas data is only stored in
# census tracts.
TRACT_TO_ZIP_FILE = (
        '/Users/nataliecygan/Desktop/Stanford/cs221/project/data/TRACT_ZIP_122015.xlsx')

""" Pickle files """
FOOD_ATLAS_PICKLE = (
        '/Users/nataliecygan/Desktop/Stanford/cs221/project/data/pickle/food_atlas.pickle')
TRACT_TO_ZIP_PICKLE = (
        '/Users/nataliecygan/Desktop/Stanford/cs221/project/data/pickle/tract_to_zip.pickle')
TRACT_TO_ZIP_MAP_PICKLE = (
        '/Users/nataliecygan/Desktop/Stanford/cs221/project/data/pickle/tract_to_zip_map.pickle')
LABELS_PICKLE = (
        '/Users/nataliecygan/Desktop/Stanford/cs221/project/data/pickle/labels.pickle')

"""
Reads in the US food access atlas data and the census tract to zipcode lookup
table. Saves both of these as .pickle files for faster loading.
"""
def read_food_access_atlas():
    tract_to_zip_data = pd.read_excel(TRACT_TO_ZIP_FILE, sheet_name=0)
    print('Read in census tract to zip lookup table, size:', tract_to_zip_data.shape)

    with open(TRACT_TO_ZIP_PICKLE, 'wb') as handle:
        pickle.dump(tract_to_zip_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved tract data to pickle file')

    atlas_data = pd.read_excel(FOOD_ATLAS_FILE, sheet_name=2)
    print('Read in food atlas data, size:', atlas_data.shape)

    with open(FOOD_ATLAS_PICKLE, 'wb') as handle:
        pickle.dump(atlas_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved atlas data to pickle file')

"""
Will read & return the pandas tables for the tract to zip data and the food
access data atlas.
"""
def read_data_frames():
    tract_df = None
    atlas_df = None

    with open(FOOD_ATLAS_PICKLE, 'rb') as fp:
        atlas_df = pickle.load(fp)

    with open(TRACT_TO_ZIP_PICKLE, 'rb') as fp:
        tract_df = pickle.load(fp)

    return tract_df, atlas_df

"""
Will read & return the pandas tables for the tract to zip data and the food
access data atlas.
"""
def read_tract_to_zip_map():
    tract_to_zip_map = None

    with open(TRACT_TO_ZIP_MAP_PICKLE, 'rb') as fp:
        tract_to_zip_map = pickle.load(fp)

    return tract_to_zip_map

"""
Given the tract to zip data frame, this function creates and returns a map of
tracts to zipcodes. This map is also saved to a .pickle file specified by the
TRACT_TO_ZIP_MAP_PICKLE constant.
"""
def create_tract_to_zip_map(tract_df):

    # Map tract --> zipcode.
    tract_to_zip = {}

    df_gen = tract_df.iterrows()

    print('Building census tract to zipcode map...')
    for i in trange(len(tract_df.index)):
        _, row = next(df_gen)

        tract = row['TRACT']
        zipcode = int(row['ZIP'])

        if tract not in tract_to_zip:
            tract_to_zip[tract] = zipcode

    with open(TRACT_TO_ZIP_MAP_PICKLE, 'wb') as handle:
        pickle.dump(tract_to_zip, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved tract to zip map to pickle file')

    return tract_to_zip

"""
Process the data from the food access atlas given the census tract to zipcode
lookup map.
This function will create a label for every zipcode and store it in a dict that
maps zipcode to label.
Ex. data[zipcode] = 0 --> zipcode is not a food desert
    data[zipcode] = 1 --> zipcode is a food desert

This dict is returned by the fuction, but also saved to a .pickle file, where
the path is specified by the LABELS_PICKLE constant.
"""
def process_data(atlas_df, tract_to_zip):

    # Map zipcode --> (number of tracts, number of food desert tracts)
    counts = {}

    df_gen = atlas_df.iterrows()
    not_found = [] # Store all tracts that cannot be looked up in tract_to_zip

    print('Processing food access atlas data to convert tracts to zipcode and \
            create labels for them...')
    for i in trange(len(atlas_df.index)):
        _, row = next(df_gen)

        # Convert census tract to zip code.
        tract = row['CensusTract']
        zipcode = None
        if tract in tract_to_zip:
            zipcode = str(tract_to_zip[tract])
        else:
            # If we can't map to a zip, just skip this entry.
            not_found.append(tract)
            continue

        # Low income & low access tract measured @ 1 mile for urban areas,
        # 10 miles for rural areas.
        LILA_1_10 = int(row['LILATracts_1And10'])
        # Low income & low access tract measured @ 1/2 mile for urban areas,
        # 10 miles for rural areas.
        LILA_half_10 = int(row['LILATracts_1And10'])
        # Low income & low access tract measured @ 1 mile for urban areas, 
        # 20 miles for rural areas.
        LILA_1_20 = int(row['LILATracts_1And10'])

        flag = LILA_1_10 or LILA_half_10 or LILA_1_20

        # If it's not already in the dict, just put down a new entry.
        if zipcode not in counts:
            counts[zipcode] = (1, flag) # If it's already there, combine with new data.  
        else: 
            previous = counts[zipcode] 
            new_count = previous[0] + 1
            flag_sum = previous[1] + flag
            counts[zipcode] = (new_count, flag_sum)

    # Maps zipcode --> 0 or 1 label indicating food desert
    data = {}
    deserts = 0

    print('Finalizing labels...')
    zipcodes = list(counts.keys())
    for i in trange(len(zipcodes)):
        # Obtain the zipcode and its corresponding counts entry.
        zipcode = zipcodes[i]
        count = counts[zipcode]
        # Divide number of food desert tracts by total tracts.
        percentage = count[1] / count[0]
        label = 1 if percentage > FOOD_DESERT_THRES else 0
        if label == 1: deserts += 1

        # Set label in 
        data[zipcode] = label

    print('Processed', len(data.keys()), 'labels.')
    print('Could not process', len(not_found), 'census tracts.')

    print('Total of', deserts, 'food deserts.')

    with open(LABELS_PICKLE, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved labels data to pickle file')

    return data

def main():
    # read_food_access_atlas()
    tract_df, atlas_df = read_data_frames()

    # Build dict from tract --> zip data
    """ Note: create_tract_to_zip_map will process directly for data frame,
    saves it to pickle. read_tract_to_zip_map will read this from .pickle"""
    # tract_to_zip = create_tract_to_zip_map(tract_df)
    tract_to_zip = read_tract_to_zip_map()

    data = process_data(atlas_df, tract_to_zip)

    print("\nSuccess")

if __name__== "__main__":
  main()
