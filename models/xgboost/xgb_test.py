import sys
from collections import defaultdict
import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
import scipy
import xgboost as xgb

from utils.utils import *

file_train = sys.argv[1]
file_test = sys.argv[2]

print(f"Loading training data from {file_train} and test data from {file_test}")

X_test, Y_test = load_svmlight_file(f"{file_test}")
X_test = xgb.DMatrix(X_test)
Y_test = np.mean(Y_test.reshape((1128, -1)), axis=0)

dtrain = xgb.DMatrix(f"{file_train}?format=libsvm#cacheprefix")

print(f"Training model")
params = {
    "eval_metric" : "mae",
    "objective" : "reg:squarederror",
    "learning_rate" : 0.1,
    "max_depth" : 5,
    "min_child_weight" : 3,
    "gamma" : 0,
    "subsample" : 0.8,
    "colsample_bytree" : 0.8,
    "scale_pos_weight" : 1,
    "seed" : 2024,
}

xgb_model = xgb.train(params, dtrain,
                      num_boost_round=100,
                      verbose_eval=10)

print("Validation:")
Y_pred = xgb_model.predict(X_test)
Y_pred = np.mean(Y_pred.reshape((1128, -1)), axis=0)
print(f"MSE: {np.mean(((-1/np.log(Y_pred)) - (-1/np.log(Y_test)))**2)}")
