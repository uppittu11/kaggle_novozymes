import os
from collections import defaultdict
import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import dump_svmlight_file
import scipy
import xgboost as xgb

from utils.utils import *


for dirname, _, filenames in os.walk('../../data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

###############################################################################
## Load in training and test data. Update training data with updated version ##
###############################################################################
#Load training csv and updated training csv
train = pd.read_csv("../../data/train.csv", index_col="seq_id")
train_updates = pd.read_csv("../../data/train_updates_20220929.csv", 
                            index_col="seq_id")

# Find and replace rows that have updated data
to_drop = train_updates[train_updates["tm"].isnull()].index
train.drop(index=to_drop, inplace=True)
train_updates.drop(index=to_drop, inplace=True)

to_change = train_updates.index
train.drop(index=to_change, inplace=True)
train = pd.concat([train, train_updates])

# Remove rows with an undefined pH in the training set
train.drop(train[train.pH.isnull()].index, inplace=True)

# Load in test csv
test = pd.read_csv("../../data/test.csv", index_col="seq_id")


###############################################################################
## Remove long protein sequences and pad remaining sequences                 ##
###############################################################################

pad_length = 4                  # length of left pad and minimum right pad
shingle_size = pad_length + 1   # size of shingles 

# Add a column for sequence length 
train.insert(2, "length", train.protein_sequence.str.len())
test.insert(2, "length", test.protein_sequence.str.len())

# Keep only sequences within the 95 %ile in length
cutoff_length = int(train.length.quantile(0.95))
train.drop(train[train.length > cutoff_length].index, inplace=True)


# Add a column for the Boltzmann inversion of Tm (provides a more normal 
# distribution)
train.insert(1, "exp_tm", np.e ** (-1/train.tm))


# Add column for padded sequence
train.insert(1, "padded_sequence",
             [string_pad(sequence, cutoff_length + 2*pad_length, pad_length)
              for sequence in train.protein_sequence])
test.insert(1, "padded_sequence",
             [string_pad(sequence, cutoff_length + 2*pad_length, pad_length)
              for sequence in test.protein_sequence])


# Get a list of the amino acids present in the dataset (note only the 20 
# standard are present here). 
amino_acid_list = list(sorted(
    {aa for sequence in test.protein_sequence for aa in set(sequence)}
))

print(f"Number of Shingles : {(cutoff_length + 2*pad_length - shingle_size)}")





###############################################################################
## Load amino acid data into                                                 ##
###############################################################################

amino_acid_data = pd.read_csv("../../data/aminoacids.csv")

# Drop string-based columns
amino_acid_data.drop(
        columns=["Name", "Abbr", "Molecular Formula", "Residue Formula"], 
        inplace=True)

# Remove non-standard residues
amino_acid_data = amino_acid_data[amino_acid_data.Letter.isin(amino_acid_list)]

# Create entry for the "buffer residue" , which represents the pad residue
buffer_residue = {
    "Letter": "X",
    "Molecular Weight": 0,
    "Residue Weight": 0,
    "pKa1" : np.NaN,
    "pKb2" : np.NaN,
    "pKx3" : np.NaN,
    "pl4"  : np.NaN,
    "H" : 0,
    "VSC" : 0,
    "P1" : 0,
    "P2" : 0,
    "SASA" : 0,
    "NCISC" : 0,
    "carbon" : 0,
    "hydrogen" : 0,
    "nitrogen" : 0,
    "oxygen" : 0,
    "sulfur" : 0,
}

# Renumber and remove the index column, add the buffer residue to df
amino_acid_data = amino_acid_data.reset_index().drop(columns="index")
amino_acid_data.loc[len(amino_acid_data.index)] = list(buffer_residue.values())

print("Amino acid features:")
print(" | ".join(amino_acid_data.columns))

# Convert dataframe into a dictionary with the residue letter as the key
amino_acid_data_dictionary = amino_acid_data.set_index('Letter').T.to_dict('list')
amino_acid_data_dictionary = {key: np.array(value, dtype=np.float32) 
                              for key, value 
                              in amino_acid_data_dictionary.items()}


###############################################################################
## Split training data into a training set and a validation set              ##
###############################################################################

# Get a list of the column names 
column_names = [f"{col}_{letter_num}"
                for letter_num in range(shingle_size)
                for col in amino_acid_data.drop(columns="Letter").columns]

train_split = 0.75  # fraction of full data set in the training set

# Separate features and target data
column_labels = train.drop(columns=["tm", "protein_sequence", "data_source", "exp_tm"]).columns
X = train.drop(columns=["tm", "protein_sequence", "data_source", "exp_tm"]).values
y = train["exp_tm"].values

X_test = test.drop(columns=["protein_sequence", "data_source"]).values


###############################################################################
## Create shingled training set with amino acid properties as features       ##
###############################################################################
    
# Allocate space for shingled datasets
shingled_dimensions = (X.shape[0], 
                       cutoff_length + 2*pad_length - shingle_size, 
                       (amino_acid_data.columns.shape[0]-1) * shingle_size)
X_shingled = np.zeros(shingled_dimensions, dtype=np.float32)
y_shingled = np.zeros(shingled_dimensions[:-1], dtype=np.float32)

# Create dataset for shingles
i = 0
t0 = time.time()
for entry_num, x_val, y_val in zip(range(len(y)), X, y):
    # Get attributes of sequence
    sequence = x_val[0]
    pH = x_val[1]
    length = x_val[2]

    # Get amino acid properties for all residues in sequence
    row = np.array([amino_acid_data_dictionary[amino_acid] for amino_acid in sequence])

    # Convert pH values to concentrations of non-dissociated residue
    row[:, [2,3, 4,5]] = row[:, [2,3, 4,5]] - pH
    row[:, 3] = row[:, 3] - (14-pH)
    row[:, 2:6] = 10 ** (row[:, 2:6])
    row[:, 2:6] = 1 / (row[:, 2:6] + 1)
    row = np.nan_to_num(row, nan=1.0)

    # Get a list of shingles
    row = np.array(
            [row[j:j+shingle_size].reshape(-1) 
             for j in range(row.shape[0]-shingle_size)]
    )
    
    # Add shingles to pre-allocated array
    X_shingled[entry_num] = row
    y_shingled[entry_num, :] = y_val
    
    # Update Counter
    i += 1
    print(f"At {int(100 * i/len(y))}%. ", 
          f"Runtime: {int((time.time()-t0)/60)} minutes", end="\r")
print()

print(X_shingled.shape)

print("Saving all train data")
if not os.path.exists("./train_all.svm.txt"):
    dump_svmlight_file(X_shingled.reshape(-1, X_shingled.shape[-1]), 
                       y_shingled.reshape(-1), 
                       f"train_all.svm.txt")
    np.savez_compressed("y_train.csv", y_shingled.reshape(-1))



shingled_dimensions = ((cutoff_length + 2*pad_length - shingle_size) * X_test.shape[0],
                       (amino_acid_data.columns.shape[0]-1) * shingle_size)
X_shingled = np.zeros(shingled_dimensions, dtype=np.float32)

i = 0
t0 = time.time()
for x in X_test:
    sequence = x[0]
    pH = x[1]
    length = x[2]

    row = np.array([amino_acid_data_dictionary[amino_acid] for amino_acid in sequence])
    row[:, [2,3, 4,5]] = row[:, [2,3, 4,5]] - pH
    row[:, 3] = row[:, 3] - (14-pH)
    row[:, 2:6] = 10 ** (row[:, 2:6])
    row[:, 2:6] = 1 / (row[:, 2:6] + 1)
    row = np.nan_to_num(row, nan=1.0)
    row = np.array([row[j:j+shingle_size].reshape(-1) for j in range(row.shape[0]-shingle_size)])

    X_shingled[i:i+(cutoff_length + 2*pad_length - shingle_size)] = row

    i += (cutoff_length + 2*pad_length - shingle_size)

    print(f"At {int(100 * i/shingled_dimensions[0])}%. Runtime: {int((time.time()-t0)/60)} minutes", end="\r")

print("Saving all test data")
pd.DataFrame(X_shingled.reshape(-1, X_shingled.shape[-1]), 
        columns=column_names).to_csv(f"X_test.csv")


