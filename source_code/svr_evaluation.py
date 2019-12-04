# Import pandas
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Import csv
import csv
# To do square root on MSE
from math import sqrt
# To enables split (train and test) of the data
from sklearn.model_selection import train_test_split
# Import svm
from sklearn import svm
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Read the data for SVR
df = pd.read_csv('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/GitHub Documentation/clean_train_one_percent.csv', sep=',')

# Labels are the values we want to predict
labels = np.array(df['hotel_cluster'])
# Remove the labels from the features
# axis 1 refers to the columns
df = df.drop('hotel_cluster', axis=1)

# Saving feature names for later use
df_list = list(df.columns)
# Convert to numpy array
df = np.array(df)

# Split the data into training and testing sets
train_df, test_df, train_labels, test_labels = train_test_split(
    df, labels, test_size=0.25, random_state=50)

# Create a svr regression
# Radial Basis Function Kernel
clf = svm.SVR(kernel='rbf', gamma='auto')

# Train the model using the training sets
clf.fit(train_df, train_labels)

# Predict the response for test dataset
pred = clf.predict(test_df)

# Calculate the absolute errors
errors = abs(pred - test_labels)

# Print out the mean absolute error (MAE)
print('MAE:', round(np.mean(errors), 2))
# Print out the mean squared error (MSE)
print('MSE: ', metrics.mean_squared_error(test_labels, pred))
# Print out the root mean squared error (RMSE)
print('RMSE: ', np.sqrt(metrics.mean_squared_error(test_labels, pred)))