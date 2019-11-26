from __future__ import division
import numpy as np
import math
from census import Census
from census import core
from us import states
from tqdm import trange

"""
Defines the CensusReader class that is used to pull data from the US census, and
useful functions for processing this data.

Implemented with python Census API (https://github.com/datamade/census)
Years available are 2011, 2012, 2013, 2014, 2015, 2016, 2017 (7 years)

Field lookup unique ids from from "Table Shells" file. Add 'E' to any field
unique lookup ID from this document.
"""

CENSUS_KEY = 'e72b09ee0cbc3819961b3b72701b5116d97bb1c2'

START_YEAR = 2011
END_YEAR = 2017

class CensusReader:

    def __init__(self):
        self.reader = Census(CENSUS_KEY) # Create census reader

    def getDataOverInterval(self, field_ids, zipcode, start_year, end_year):
        """
        Will retrieve data over the time interval specified by start_year and
        end_year for the given zipcode and field_id list. Will return a list 
        where each element is the data value for each year in the specified
        range as numpy array.
        If field_ids contains multiple fields, this will combine the numbers
        into one result. This is useful for the census fields that are divided
        by sex, race, etc. Additionally it is useful when looking for the union
        of the categories (ex. college education would require several fields).
        """
        values = []
        for y in range(start_year, end_year + 1):
            res = None
            while True:
                try:
                    res = self.reader.acs5.zipcode(field_ids, zipcode, year=y)
                    break # If we get here, there was no exception.
                except core.CensusException:
                    pass # Just try again until we get a connection.

            # if this is empty, return -1 to indicate this.
            if len(res) == 0: return -1
            value = sum([int(res[0][key]) for key in field_ids])

            values.append(value)
        return np.array(values)

    def normalizeToPopulation(self, population, raw_data):
        """
        Given a raw_data vector of individual counts, this function will represent
        that number as a percentage.
        """
        assert(population.shape == raw_data.shape)
        percentages = raw_data / population
        return percentages

