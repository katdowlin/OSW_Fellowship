# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:58:44 2024

@author: kevin
"""
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



scada_data = pd.read_csv(r"C:\Users\kevin\Downloads\sumr-d-5-min-database (2).xlsx - Means (1).csv")
#scada_data = pd.read_csv(r"C:\Users\kevin\Downloads\4_variable_scada.csv")

scada_data.describe()

# Extract feature and target arrays
X, y = scada_data.drop('isOperate', axis=1), scada_data[['isOperate']]
             # Define target
# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle = False)

# Identify categorical features (if any)
categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

# Train CatBoost model
model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)
model.fit(X_train, y_train, cat_features=categorical_features_indices)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

import numpy as np

rounded_preds = np.round(y_pred).astype(int)


# Convert to NumPy arrays if needed
y_test = np.array(y_test).flatten()
y_pred = np.array(rounded_preds).flatten()

# Check shapes again
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of y_pred: {y_pred.shape}")


# Calculate errors
errors = y_test - y_pred

import matplotlib.pyplot as plt

# Plot fitted values vs. errors
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, errors, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Fitted Values vs. Prediction Errors")
plt.xlabel("Fitted Values (Predictions)")
plt.ylabel("Errors (True - Predicted)")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='green', alpha=0.7)
plt.title("Error Distribution for Catboost Model")
plt.xlabel("Errors (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression
import numpy as np

# Extract feature importances from CatBoost
feature_importances = model.feature_importances_
feature_names = X.columns

# Print feature importances
for name, importance in zip(feature_names, feature_importances):
    print(f"Feature: {name}, Importance: {importance:.2f}")
    

import matplotlib.pyplot as plt
import pandas as pd

# Get feature importance and convert to DataFrame
catboost_importance = model.get_feature_importance(prettified=True)
catboost_importance_df = pd.DataFrame(catboost_importance)

# Sort by importance and select top 10
top_catboost_features = catboost_importance_df.nlargest(10, 'Importances')

# Plot top 10 features
plt.figure(figsize=(10, 6))
plt.barh(top_catboost_features['Feature Id'], top_catboost_features['Importances'], color='skyblue', edgecolor='black')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 CatBoost Feature Importance")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# Fit a linear regression model using the CatBoost predictions
linear_model = LinearRegression()
linear_model.fit(X_train, model.predict(X_train))

# Get the coefficients and intercept
coefficients = linear_model.coef_
intercept = linear_model.intercept_

# Print the linear equation
equation = " + ".join(
    [f"{coef:.3f}*{name}" for coef, name in zip(coefficients, feature_names)]
)
print(f"Linear Equation: y = {intercept:.3f} + {equation}")
