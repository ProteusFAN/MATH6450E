#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# Data Loading
submission = pd.read_csv('./Data/submission.csv')
TrueIC50s = pd.read_csv('./Data/TrueIC50s.csv')
df = pd.read_csv('./Data/OneDrug.csv')
train_index = df.index[np.logical_not(np.isnan(df['IC50s']))]
test_index = df.index[np.isnan(df['IC50s'])]
train_feature = df.values[train_index,3:]
train_IC50s = df.values[train_index,2]
test_feature = df.values[test_index,3:]
test_IC50s = np.zeros(test_feature.shape[0])

# adaptive empirical conditional expectation method + bootstrap
from AECE import aece
from sklearn.metrics import mean_squared_error

# normal method
model = aece()
model.fit(train_feature, train_IC50s)
test_IC50s = model.predict(test_feature)

# mean of bootstrap
model = aece()
model.fit(train_feature, train_IC50s)
test_IC50s = model.predict_bootstrap(test_feature)

mse = mean_squared_error(test_IC50s, TrueIC50s['IC50'])

submission['IC50'] = test_IC50s
submission.to_csv('./Data/aece_bootstrap_mean.csv', index=False)