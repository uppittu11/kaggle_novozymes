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

print(f"Loading training and test data")

dtrain = xgb.DMatrix(f"train_all.svm.txt?format=libsvm#cacheprefix")
X_test = xgb.DMatrix('X_test.csv?format=csv#cacheprefix')

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

print("Prediction:")
Y_pred = xgb_model.predict(X_test)
Y_pred = np.mean(Y_pred.reshape((1128, -1)), axis=0)
Y_pred = -1 / np.log(Y_pred)

np.savez_compressed("y_pred.npz_compressed", y_pred)

