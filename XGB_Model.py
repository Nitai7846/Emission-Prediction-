#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:59:17 2023

@author: nitaishah
"""


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("")



def select_features(df, feature_list):
    X = df[feature_list]
    return X

def select_target(df, target_list):
    y = df[target_list]
    return y

def train_test_split_scale(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return mse, mae, rmse, r2
    
    


feature_list = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']

X = select_features(df, feature_list)

target_list = ['NOX']  # Can Be set to "CO" as well
y = select_target(df, target_list)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split_scale(X, y)

model = XGBRegressor()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse, mae, rmse, r2 = calculate_regression_metrics(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)
