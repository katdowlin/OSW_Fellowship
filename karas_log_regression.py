# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:57:15 2024

@author: kevin
"""

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2



scada_data = pd.read_csv(r"C:\Users\kevin\Downloads\sumr-d-5-min-database (2).xlsx - Means.csv")

scada_data.describe()

# Extract features and target arrays
X, y = scada_data.drop('isOperate', axis=1), scada_data[['isOperate']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the model without regularization
model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=X_train.shape[1]))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Logistic regression with L1 and L2 regularization
reg = L1L2(l1=0.01, l2=0.01)
modelr = Sequential()
modelr.add(Dense(1, activation='sigmoid', kernel_regularizer=reg, input_dim=X_train.shape[1]))
modelr.compile(optimizer='rmsprop', loss='binary_crossentropy')
modelr.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Calculate MSE for the non-regularized model
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # Flatten to 1D
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")

# Calculate MSE for the regularized model
y_pred_proba_r = modelr.predict(X_test)
y_pred_r = (y_pred_proba_r > 0.5).astype(int).flatten()  # Flatten to 1D
mse_r = mean_squared_error(y_test, y_pred_r)
print(f"Mean Squared Error Reg: {mse_r:.3f}")

# Convert predictions to DataFrames before saving to CSV
y_pred_df = pd.DataFrame(y_pred, columns=["Prediction"])
y_pred_r_df = pd.DataFrame(y_pred_r, columns=["Prediction_Reg"])

# Save predictions to CSV
#y_pred_df.to_csv("y_pred.csv", index=False)
#y_pred_r_df.to_csv("y_pred_r.csv", index=False)

# Download CSV files
#files.download("y_pred.csv")
#files.download("y_pred_r.csv")

# Convert y_test to a Series and ensure it is numeric
y_test_numeric = pd.to_numeric(y_test.squeeze(), errors='coerce')  # Convert y_test to 1D Series

# Calculate errors for non-regularized model
errors = y_test_numeric - y_pred

# Plot fitted values vs. errors for non-regularized model
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, errors, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Fitted Values vs. Prediction Errors (Non-Regularized)")
plt.xlabel("Fitted Values (Predictions)")
plt.ylabel("Errors (True - Predicted)")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Histogram for errors from non-regularized model
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Prediction Errors (Non-Regularized Model)")
plt.xlabel("Error (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

logreg_coefficients = model.layers[0].get_weights()[0]
print("Non-Regularized Logistic Regression Coefficients:", logreg_coefficients)

# Get weights (coefficients) from the regularized logistic regression model
logreg_r_coefficients = modelr.layers[0].get_weights()[0]
print("Regularized Logistic Regression Coefficients:", logreg_r_coefficients)

# Calculate errors for regularized model
errors_r = y_test_numeric - y_pred_r

# Plot fitted values vs. errors for regularized model
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_r, errors_r, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Fitted Values vs. Prediction Errors (Regularized)")
plt.xlabel("Fitted Values (Predictions)")
plt.ylabel("Errors (True - Predicted)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(errors_r, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title("Histogram of Prediction Errors (Regularized Model)")
plt.xlabel("Error (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()