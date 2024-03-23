import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import dump_svmlight_file

from utils.utils import *

index_dictionary = dict()

frac_train = 0.75
n_entries = 27261
n_shingles = 1128 

# Create 3 training and validation sets 
for training_round, rand_seed in [(0, 123), (1, 456), (2, 789)]:
    # create a random number generator 
    gen = np.random.default_rng(seed=rand_seed)
    
    # Split the dataset into the train and validation sets
    indices = np.arange(n_entries)
    gen.shuffle(indices)
    train_split_idx = int(n_entries * frac_train)
    train_idxs, val_idxs = indices[:train_split_idx], indices[train_split_idx:]

    index_dictionary[f"train_{training_round}"] = set(train_idxs)
    index_dictionary[f"val_{training_round}"] = set(val_idxs)

with open("train_all.svm.txt", "r") as f_all:
    for i, line in enumerate(f_all):
        entry_num = int(i/n_shingles)
        for key, val in index_dictionary.items():
            if entry_num in val:
                with open(f"{key}.svm.txt", "a") as f_round:
                    f_round.write(line)
        if i % 10000 == 0:
            print(f"{int(100 * i / n_entries / n_shingles)}% complete", end="\r")

