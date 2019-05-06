#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Packages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from AECE import aece
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Data Loading
True_IC50s = pd.read_csv('./Data/TrueIC50s.csv')
df = pd.read_csv('./Data/OneDrug.csv')
train_index = df.index[np.logical_not(np.isnan(df['IC50s']))]
test_index = df.index[np.isnan(df['IC50s'])]
train_feature = df.values[train_index,3:]
train_IC50s = df.values[train_index,2]
test_feature = df.values[test_index,3:]
test_IC50s = np.zeros(test_feature.shape[0])

# Comparison of algorithms
mse_aece = 0
mse_aece_bs = 0
mse_gb = 0
mse_rf = 0
mse_svr = 0
mse_mlp = 0

# Adaptive Empirical Conditional Expectation
model = aece()
model.fit(train_feature, train_IC50s)
mse_ece = mean_squared_error(model.predict_naive(test_feature), True_IC50s)
mse_aece = mean_squared_error(model.predict(test_feature), True_IC50s)
mse_aece_bs = mean_squared_error(model.predict_bootstrap(test_feature), True_IC50s)
    
# Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
      'learning_rate': 0.01, 'loss': 'ls'}
gb = GradientBoostingRegressor(**params)
gb.fit(train_feature, train_IC50s)
mse_gb = mean_squared_error(gb.predict(test_feature), True_IC50s)
    
# Random Forest
rf = RandomForestRegressor(n_estimators=50, max_depth=30)
rf.fit(train_feature, train_IC50s)
mse_rf += mean_squared_error(rf.predict(test_feature), True_IC50s)
    
# SVR
svr = SVR(C=1.0, epsilon=0.2)
svr.fit(train_feature, train_IC50s)
mse_svr = mean_squared_error(svr.predict(test_feature), True_IC50s)
    
# MLP
mlp = MLPRegressor(hidden_layer_sizes=(60,60), early_stopping=True)
mlp.fit(train_feature, train_IC50s)
mse_mlp = mean_squared_error(mlp.predict(test_feature), True_IC50s)
    