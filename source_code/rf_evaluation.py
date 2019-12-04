# Import pandas
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Import csv
import csv
# To do square root on MSE
from math import sqrt
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Import MSE & RMSE
from sklearn import metrics

# Read in data and display first 5 rows
features = pd.read_csv('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/GitHub Documentation/csv_data/clean_train.csv', sep=',')

# Labels are the values we want to predict
labels = np.array(features['hotel_cluster'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('hotel_cluster', axis = 1)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Instantiate model with 1 decision trees
rf = RandomForestRegressor(n_estimators = 1, random_state = 42)
# Train the model on training data  
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (MAE)
print('MAE:', round(np.mean(errors), 2))
# Print out the mean squared error (MSE)
print('MSE: ', metrics.mean_squared_error(test_labels, predictions))
# Print out the root mean squared error (RMSE)
print('RMSE: ', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
# Print out the coefficient of determination (R^2)
print('R*R:', rf.score(test_features, test_labels))