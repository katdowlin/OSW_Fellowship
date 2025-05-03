# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:52:23 2024

@author: kevin
"""

import pandas as pd
col_names = [
    'isOperate', 'Teeter_brake_pressure', 'Pitch_digital_outputs', 'Blade_1_pitch_RPM', 
    'Teeter_Angle', 'Blade_1_pitch_current', 'Blade_1_pitch_angle', 'Blade_2_pitch_RPM', 
    'Blade_2_pitch_angle', 'Blade_2_pitch_current', 'LSS_torque', 'Blade_1_edge_bending', 
    'Blade_1_flap_bending', 'Blade_2_edge_bending', 'Blade_2_flap_bending', 
    'Blade_1_13m_edge_bending', 'Blade_1_13m_flap_bending', 'Blade_2_13m_edge_bending', 
    'Blade_2_13m_flap_bending', 'Gearbox_oil_temperature', 'HSS_position', 'Yaw_position', 
    'HSS_torque', 'LSS_position', 'Yaw_brake_pressure', 'Nacelle_digital_outputs', 
    'X_Accelerometer_Port', 'Y_Accelerometer_Port', 'Z_Accelerometer_Port', 
    'X_Accelerometer_Starboard', 'Y_Accelerometer_Starboard', 'Z_Accelerometer_Starboard', 
    'Nacelle_wind_speed', 'Nacelle_wind_direction', 'Gearbox_oil_pressure', 
    'Nacelle_ambient_temperature', 'IMU_Roll', 'IMU_Pitch', 'IMU_Yaw', 'IMU_X', 'IMU_Y', 
    'IMU_Z', 'Generator_Power', 'PE_digital', 'Generator_Current', 'GENFREQ', 
    'Generator_Voltage', 'PE_Line_power', 'PE_kVA', 'PE_Power_Factor', 'Tower_bending_E_Slash_W', 
    'Tower_bending_N_Slash_S', 'Tower_bending_E_Slash_W_Poisson', 'Tower_bending_N_Slash_S_Poisson', 
    'Met_wind_speed_58p2m', 'Met_wind_direction_58p2m', 'Met_temperature_58p2m', 
    'Met_wind_speed_36p6m', 'Met_wind_direction_36p6m', 'Met_wind_speed_15m', 
    'Met_wind_direction_15m', 'Met_wind_speed_3m', 'Met_temperature_3m', 'Met_sonic_U', 
    'Met_sonic_V', 'Met_sonic_W', 'Watchdog_RPM', 'Barometric_pressure', 'GPS_Year', 
    'GPS_Month', 'GPS_Day', 'GPS_Hour', 'GPS_Minute', 'GPS_Second', 'GPS_Tenths_of_milliseconds', 
    'GPS_Validity', 'HSS_RPM', 'LSS_RPM', 'Air_Density', 'TSR', 'HSS_Power', 'LSS_Power', 
    'LSS_omega_dot', 'HSS_omega_dot', 'Set_Pitch_1', 'Set_Pitch_2', 'Blade_1_pitch_rate', 
    'Blade_2_pitch_rate', 'Blade_1_pitch_accel', 'Blade_2_pitch_accel', 
    'Blade_1_pitch_rate_command_2_pole', 'Blade_2_pitch_rate_command_2_pole', 
    'Blade_1_pitch_rate_error_criteria_2_pole', 'Blade_2_pitch_rate_error_criteria_2_pole', 
    'Blade_1_pitch_current_2_pole', 'Blade_2_pitch_current_2_pole', 'StreamTime0', 
    'StreamTime1', 'StreamTime2', 'StreamTime3'
]

scada_data = pd.read_csv(r"C:\Users\kevin\Downloads\sumr-d-5-min-database (2).xlsx - Means (1).csv", 
                         header = None, names = col_names)

scada_data = scada_data.drop(scada_data.index[0])
                         
scada_data.head()                         

X = scada_data.drop('isOperate', axis=1)

y = scada_data['isOperate']

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# import the class
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression on scaled data
model = LogisticRegression(max_iter=1000, random_state=16)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


log_reg_coeff = model.coef_

print("Model Coefficients: ", model.coef_)
print("Model Intercept: ", model.intercept_)
print("Model Parameters: ", model.get_params())
accuracy = model.score(X_test_scaled, y_test)
print("Model Accuracy: ", accuracy)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.3f}")

#y_pred.to_csv("y_pred.csv", index=False)


# Ensure y_test and y_pred are numeric
y_test = pd.to_numeric(y_test, errors='coerce')
y_pred = pd.to_numeric(y_pred, errors='coerce')

# Calculate errors
errors = y_test - y_pred

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
plt.title("Error Distribution for Non-Regularized Logistic Regression Model")
plt.xlabel("Errors (True - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



