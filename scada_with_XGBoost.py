# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:21:49 2024

@author: kevin
"""

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings


warnings.filterwarnings("ignore")


scada_data = pd.read_csv(r"C:\Users\kevin\Downloads\sumr-d-5-min-database (2).xlsx - Means (1).csv")

#scada_data = pd.read_csv(r"C:\Users\kevin\Downloads\4_variable_scada.csv")

scada_data.describe()

from sklearn.model_selection import train_test_split

# Extract feature and target arrays
X, y = scada_data.drop('isOperate', axis=1), scada_data[['isOperate']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle = False)

import xgboost as xgb

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

n = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)


from sklearn.metrics import mean_squared_error

preds = model.predict(dtest_reg)

mse = mean_squared_error(y_test, preds, squared=True)

print(f"MSE of the base model: {mse:.3f}")

params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]

n = 10000


model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=5,
   # Activate early stopping
   early_stopping_rounds=50
)


import pandas as pd

import xgboost as xgb
from xgboost import plot_importance



# Plot importance with max_num_features set to 10
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=10, importance_type='weight', show_values=False)
plt.title("Top 10 XGBoost Feature Importance")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()



# Convert the NumPy array to a DataFrame
preds_df = pd.DataFrame(preds)

# Save to CSV
#preds_df.to_csv("preds.csv", index=False)

rounded_preds = np.round(preds).astype(int)

# Convert y_test to a Series and ensure it is numeric
y_test_numeric = pd.to_numeric(y_test.squeeze(), errors='coerce')  # Squeeze to make it a 1-dimensional Series

# Calculate errors
errors = y_test_numeric - rounded_preds

# Plot fitted values vs. errors
plt.figure(figsize=(10, 6))
plt.scatter(rounded_preds, errors, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Fitted Values vs. Prediction Errors")
plt.xlabel("Fitted Values (Predictions)")
plt.ylabel("Errors (True - Predicted)")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Histogram for errors in XGBoost model
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='blue', alpha=0.7)
plt.title("Error Distribution for XGBoost Model")
plt.xlabel("Errors (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

xgb_feature_importances = model.get_score(importance_type='weight')
xgb_coefficients = np.array(list(xgb_feature_importances.values()))

print("XGBoost Feature Importances (as coefficients):", xgb_coefficients)

from sklearn.linear_model import LinearRegression

# Get feature importances from XGBoost
feature_importances = model.get_score(importance_type='weight')
feature_names = list(feature_importances.keys())
coefficients = np.array(list(feature_importances.values()))

# Normalize feature importances to create relative weights
normalized_coefficients = coefficients / coefficients.sum()

# Fit a linear regression model to XGBoost predictions
linear_model = LinearRegression()
linear_model.fit(X_train, model.predict(dtrain_reg))  # Fit using XGBoost predictions

# Get linear regression coefficients and intercept
linear_coefficients = linear_model.coef_
intercept = linear_model.intercept_

# Print feature importances
print("XGBoost Feature Importances:")
for name, importance in zip(feature_names, normalized_coefficients):
    print(f"Feature: {name}, Normalized Importance: {importance:.4f}")

# Print the approximate linear equation
equation = " + ".join(
    [f"{coef:.3f}*{name}" for coef, name in zip(linear_coefficients, X.columns)]
)
print(f"Approximate Linear Equation: y = {intercept:.3f} + {equation}")




