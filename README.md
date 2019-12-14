# Predicting Food Deserts in the United States: a CS 221 Project
Authors: Natalie Cygan, Grace Kim, Priscilla Lui. 

---

**Abstract**: Food Deserts are regions of the United States with limited access to affordable and nutritious food. Aimed at helping policy-makers and produce-providers understand where to best focus their efforts to alleviate food security, we built a predictive model to classify areas as food deserts or not using socioeconomic features and identified which of these features were the best predictors of food deserts. The best model was a neural network that achieved f-1 scores of 0.72 for both classes.

## Model Instructions

First, install all requirements:
```
pip install -r requirements.txt
```

Different classifier models can be run with `python <name of model>`, where all classifiers exist in the `models/` directory. For specific model directions, please refer to their respective `.py` files.

## Data Instructions

The `data_processing/` directory contains our suite of tools we create to build our joint dataset from Census features and USDA Food Accesss Research Atalas labels. Please refer to the documentation present in those files to to understand how to use them.

Our fully cleaned dataset is present as a `.pickle` file in `data/full_data.pickle`. It is stored as python dict where zip code key values are mapped to their respective data point, which are represented as `(feature vector, label)` pairs.
