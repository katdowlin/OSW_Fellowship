# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:11:40 2024

@author: kevin
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


scada_data = pd.read_csv(r"C:\Users\kevin\Downloads\sumr-d-5-min-database (2).xlsx - Means (1).csv")
#scada_data = pd.read_csv(r"C:\Users\kevin\Downloads\4_variable_scada.csv")

scada_data.describe()

# Extract feature and target arrays
X, y = scada_data.drop('isOperate', axis=1), scada_data[['isOperate']]
             # Define target
# Split into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle = False)

# Initialize LightGBM classifier
model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42, verbose=-1)

# Fit the model with evaluation sets
model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test), (x_train, y_train)],
    eval_metric='logloss'
)

from sklearn.metrics import mean_squared_error

# Predict on the test set
y_pred = model.predict(x_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")

# Evaluate training and testing accuracy
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}")

# Feature Importance
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance")
plt.show()

# Validation Metric Plot
plt.figure(figsize=(10, 6))
lgb.plot_metric(model)
plt.title("Validation Metrics Over Iterations")
plt.show()


preds_class_0 = y_pred[y_test == 0]
preds_class_1 = y_pred[y_test == 1]

# Plot
plt.figure(figsize=(8,6))
plt.hist(preds_class_0, bins=25, alpha=0.6, label='Class 0', color='skyblue', edgecolor='black')
plt.hist(preds_class_1, bins=25, alpha=0.6, label='Class 1', color='salmon', edgecolor='black')

plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities by Class (LightGBM)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (LightGBM)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

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
plt.title("Error Distribution for LightGBM Model")
plt.xlabel("Errors (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression

# Extract feature importances from LightGBM
feature_importances = model.feature_importances_
feature_names = X.columns

# Print feature importances
for name, importance in zip(feature_names, feature_importances):
    print(f"Feature: {name}, Importance: {importance:.2f}")

# Fit a linear regression model using the LightGBM predictions
linear_model = LinearRegression()
linear_model.fit(x_train, model.predict_proba(x_train)[:, 1])  # Use probability estimates for fitting

# Get the coefficients and intercept
coefficients = linear_model.coef_
intercept = linear_model.intercept_

# Print the linear equation
equation = " + ".join(
    [f"{coef:.3f}*{name}" for coef, name in zip(coefficients, feature_names)]
)
print(f"Approximate Linear Equation: y = {intercept:.3f} + {equation}")

