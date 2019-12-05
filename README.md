# Expedia Hotel Recommendation System
This repository is created to explain our project on Frontier Technology 2019 (Data Science) on our major, Informatics. The goal of this software is to predict which hotel group (hotel clusters) a user is going to book based on the given input. This software will display a map together with the hotel recommendation. Therefore, as the train data from the year 2013 and 2014 cannot be opened, the data that we will use here is from the test.csv which is year 2015. This software will need an input of user's detail and then it will proceed in processing the data and give outcome from some machine learning model.

## Authors
Alessandro Luiz Kartika (01082170029)<br>
Denny Raymond (01082170017)<br>
Winston Renatan (01082170030)<br>
Informatics 2017, Universitas Pelita Harapan Main Campus

## Requirement
First we need to download and install the development environment as mentioned below:<br>
[Orange](https://orange.biolab.si/download/#windows)<br>
Guide on how to install Orange: https://youtu.be/XDdi978Xk7Y <br>
[Python 3.7](https://www.python.org/downloads/)<br>
Guide on how to download and install Python: https://youtu.be/dX2-V2BocqQ <br>
[Jupyter Notebook](https://jupyter.org/install) <br>
Besides the Python itself, we also need some package to import such as below:<br>
[Pandas](https://pandas.pydata.org/)<br>
[NumPy](https://numpy.org/)<br>
[scikit-learn](https://scikit-learn.org/stable/)<br>
Or we can just simply download the whole thing in [Anaconda](https://www.anaconda.com/distribution/).<br>
Don't forget to also to install [Dash](https://dash.plot.ly/installation) for the display.<br>

## Understanding the Data
### train and test.csv
The data in test.csv gives us information of user behavior that is logged. Here, it includes details what customer searches are and how customer made interaction with the results. For example, is the hotel part of package, are they really do booking on hotel or just click by, and others. Our goal is to predict the **hotel clusters** based on the given data from Expedia and user input later on.

| Feature Name              | Description                                              |
| :---:                     | :---:                                                    |
| ID                        | denotes the ID                                           |
| date_time                 | timestamp                                                |
| site_name                 | ID of Expedia point of sales                             |
| posa_continent            | ID of continent associate with site name                 |
| user_location_country     | ID of the user country location                          |
| user_location_region      | ID of the user region location                           |
| user_location_city        | ID of the user city location                             |
| orig_destination_distance | distance of hotel and customer (when doing search)       |
| user_id                   | ID of the user                                           |
| is_mobile                 | 1: access from mobile device, 0: others                  |
| is_package                | 1: booking part of package, 0: others                    |
| channel                   | ID of marketing channel                                  |
| srch_ci                   | check-in                                                 |
| srch_co                   | check-out                                                |
| srch_adults_cnt           | number of adults in hotel room in search                 |
| srch_children_cnt         | number of children in hotel room in search               |
| srch_rm_cnt               | number of rooms wanted in search                         |
| srch_destination_id       | ID of destination where hotel search was performed       |
| srch_destination_type_id  | type of destination                                      |
| hotel_continent           | hotel continent                                          |
| hotel_country             | hotel country                                            |
| hotel_market              | hotel market                                             |
| cnt                       | number of similar events in context of same user session |
| is_booking                | 1: is booking, 0: is click                               |
| hotel_cluster             | ID of hotel cluster **prediction**                       |

### destinations.csv
This data contains information about the hotel reviews made by users and extracted as a features. But, on this project we will not use this information. <br>

| Feature Name              | Description                                              |
| :---:                     | :---:                                                    |
| srch_destination_id       | ID of destination where hotel search was performed       |
| d1-d149                   | latent description of search regions                     |

## Implementing Machine Learning
### Implementation Steps
![ImplementationWorkFlow](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/project_workflow.PNG)<br>

### Data Acquisition
This part will start from entering kaggle Expedia Hotel Recommendations and download the data on [Here](https://www.kaggle.com/c/expedia-hotel-recommendations/data). Then we put it to the folder we want to work with.<br>

### Data Cleansing
Here what we wanted to do is to clean the data so that it is more compact and clear. The cleaning here will use clean_data.py on the folder source_code. First we need to import pandas that enables us to work with csv files. <br>
```Python
# Import pandas
import pandas as pd
```
First of all, we should do is to read the file and put our file location there. The file meant is train.csv which more or less have the size of 4GB. We also store the csv files to a dataframe. The read function also includes to drop rows which contain `null` in its column. <br>
![NoValueOnOrigDestinationDistance](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/orig_destination_distance_value_nan.png)<br>
```Python
# read the train file to dataframe and drop rows that have no value.
df = pd.read_csv ('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/first_try/raw_data/train.csv', sep=',').dropna()
```
The next thing is that, we would like to focus just on the data where is_booking value is 1. Why? it is because when is_booking value equals to 0, people just seeing through (window shopping) but not booking or doing the transaction.<br>
![IsBookingValue](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/is_booking_value_one.png)<br>
```Python
# delete all rows with 0 value at is_booking
df = df[(df.is_booking != 0)]
```
Besides, we would also like to cut the column for date_time, srch_ci, and srch_co. As the algorithm we provide cannot work with strings and dates.<br>
![DateValue](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/date_value_string.png)<br>
The code in python is written below.<br>
```Python
# delete all columns below as it is presented as string and cannot be used in our algorithm
df = df.drop('is_booking', axis = 1)
df = df.drop('date_time', axis = 1)
df = df.drop('srch_ci', axis = 1)
df = df.drop('srch_co', axis = 1)
```
The last thing we should do is to store the data in a new csv file after doing the cleansing. Remember to change the file location as yours. <br>
```Python
# write the cleaned dataframe result to a new file.
export_csv = df.to_csv ('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/first_try/clean_train.csv', index = None, header=True)
```

### Orange Evaluation
Here we would like to see which algorithm or model that works with our data. We will evaluate using the data that have been cleaned and also once randomize it just using the 1% of the data (19855 data) so that Orange can run smooth on my machine. To cut the data for Orange, we can use the minimize_file.py. On the code below, it is stated that we only took 1% (0,01) and the random_state is to initialize the random function. <br>
```Python
# .sample = Return a random sample of items from an axis of object.
# frac = Fraction of axis items to return.
# random_state = Mersenne Twister pseudo random number generator.
df = df.sample (frac=0.01, random_state=99)
# see the shape (rows and columns) of the data (exclude header).
print(df.shape)
```
The result below shows the result of using Orange, that our data works with Random Forest and Support Vector Machine, thus that is going to be implemented.<br>
![OrangeVideo](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/orange_workflow.gif)<br>

The detailed evaluation result of the SVM and RF on RMSE, MSE, MAE, and R<sup>2</sup> according to Orange.<br>
![OrangeEvaluation](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/orange_test_and_score.PNG)<br>

MAE (Mean Absolute Error) measures the average magnitude of error in set of predictions. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.
Here is the mathematical equation for MAE: ![MAE](https://latex.codecogs.com/gif.latex?MAE%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Cp_%7Bi%7D-a_%7Bi%7D%29%7C%7D%7Bn%7D)<br>

RMSE (Root Mean Squared Error) is quadratic scoring rule that measures average magnitude of error. <br>
Following is the equation of MSE together with RMSE:<br>
![MSE](https://latex.codecogs.com/gif.latex?MSE%3D%5Cfrac%7B1%7D%7Bn%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_%7Bi%7D-%5Coverline%7By%7D_%7Bi%7D%29%5E%7B2%7D%7D)<br>
![RMSE](https://latex.codecogs.com/gif.latex?RMSE%3D%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28p_%7Bi%7D-a_%7Bi%7D%29%5E%7B2%7D%7D%7Bn%7D%7D)<br>

Besides, we also have the R<sup>2</sup> which measures the strength and direction of a linear relationship between two variables. The main goal is to get as close to -1 or +1. Where exactly –1 means its a perfect downhill (negative) linear relationship and on the other side exactly +1 means a perfect uphill (positive) linear relationship. With the following mathematical equation:
![RSquared](https://latex.codecogs.com/gif.latex?R%5E%7B2%7D%3D1-%5Cfrac%7BExplained%20Variation%7D%7BTotal%20Variation%7D)<br>

Meanwhile, we also can see which information give the most impact to the prediction result. As below, the highest five are user_id, user_location_city, hotel_market, srch_destination_id, and user_location_region. So as we can conclude, the three of the top five rank is on the user details.<br>
![OrangeRank](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/orange_rank_top_five.png)<br>

### Model Train and Test
Here we work with two models, which is Random Forest (RF) and Support Vector Regression (SVR).<br>
Random Forest itself builds upon the idea of bagging which reduces overfiting. This model have two parameters which are, the number of trees and the number of features. The error will depends on correlation between trees and strength of each single tree. This method is also easy to parallelize, that makes it good for use. Here using the rf_evaluation.py and for this model we use 100% of the cleaned data. Before starting the evaluation process, we need to import some things as explained below. <br>
```Python
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
```
We continue to read the data that we have cleaned and setting labels (target) as hotel_cluster and drop the column there. After dropping it we should also convert the result to a numpy array for further process. The code is shown below. <br>
```Python
# Read in data and display first 5 rows
features = pd.read_csv('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/GitHub Documentation/clean_train.csv', sep=',')

# Labels are the values we want to predict
labels = np.array(features['hotel_cluster'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('hotel_cluster', axis = 1)
# Convert to numpy array
features = np.array(features)
```
We proceed to the next step by spiliting the data to train and test. Where in the training sets consists of 75% of the total data and 25% as the test set. random_state is just being used to initialize the random function on the train_test_split process.<br>
```Python
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
```
After doing the split, we can now train the data and predict the test sets. The codes shown below that we initiate the Random Forest with one decision tree (n_estimators), when the n_estimators increase it will get us to a more detailed number in decimals. Thus, we choose it as 1 rather than the others.<br>
```Python
# Instantiate model with 1 decision trees
rf = RandomForestRegressor(n_estimators = 1, random_state = 42)
# Train the model on training data  
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
```
After doing the train and prediction, now it is the time to evaluate how our model works. What we want to see is its MAE, MSE, RMSE, and R<sup>2</sup>. <br>
```Python
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
```

For the evaluation we get the result as below. <br>

|Evaluation        |Value                     |
|------------------|--------------------------|
|MAE               |28.2                      |
|MSE               |1430.1012903022558        |
|RMSE              |37.81668005394254         |
|R<sup>2</sup>     |-0.6820501555223537       |

Support Vector Regression can both be applied to solve problems in classification and regression. It also able to endure with multiple variables. There are many methods (kernel we say) to work with such as Linear, Polynomial, Radial Bias Function, and much more. This model gives out good accuracy and use less memory. On the downside, SVR comes with long training time for large dataset. Here using the svr_evaluation.py for the evaluation we get the result as below. For this model we only use 1% of the cleaned data, because for some tries on our machines it took a long time to run the data in bigger percentage. With almost the same exact steps as before, there are some things that we need to import as mentioned below. <br>
```Python
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
```
The first thing that we must to do is to read the data, we also would like to select the hotel_cluster as our target result and drop the data. After dropping the data we would like to convert the data to numpy array.<br>
```Python
# Labels are the values we want to predict
labels = np.array(df['hotel_cluster'])
# Remove the labels from the features
# axis 1 refers to the columns
df = df.drop('hotel_cluster', axis=1)
# Convert to numpy array
df = np.array(df)
```
We continue to split the data to train and test, where the train data would be 75% and test would be 25%. <br>
```Python
# Split the data into training and testing sets
train_df, test_df, train_labels, test_labels = train_test_split(
    df, labels, test_size=0.25, random_state=50)
```
The process continue with creating an svr regression with the Radial Basis Function as the kernel. We proceed with training the data using the training sets that we have split before. The next thing we would like to do is to predict the data on the test set.<br>
```Python
# Create a svr regression
# Radial Basis Function Kernel
clf = svm.SVR(kernel='rbf', gamma='auto')

# Train the model using the training sets
clf.fit(train_df, train_labels)

# Predict the response for test dataset
pred = clf.predict(test_df)
```
Last thing we would like to know is its evaluation, with calculating the errors and having MAE, MSE, and RMSE to evaluate the model. <br>
```Python
# Calculate the absolute errors
errors = abs(pred - test_labels)

# Print out the mean absolute error (MAE)
print('MAE:', round(np.mean(errors), 2))
# Print out the mean squared error (MSE)
print('MSE: ', metrics.mean_squared_error(test_labels, pred))
# Print out the root mean squared error (RMSE)
print('RMSE: ', np.sqrt(metrics.mean_squared_error(test_labels, pred)))
```
For the evaluation we get the result as below. <br>

|Evaluation        |Value                     |
|------------------|--------------------------|
|MAE               |24.55                     |
|MSE               |845.3880810817358         |
|RMSE              |29.307555813878275        |

From both result we can compare that Support Vector Regression has better results as the error is lower than what is given out in the Random Forest result. For example, the MAE of Support Vector Regression 24.55 while Random Forest is larger with the result is 28.2. We can see the plot result on a Jupyter Notebook file [Here](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/source_code/SVM%20%26%20RF%20Evaluation%20Plot.ipynb).

### Dash Development
At the same time with Model Train and Test, we also would like to display our result not just in the terminal so we develop the front-end with Python Dash. This development also uses bootstrap. The thing that we should import are mentioned as below. <br>
```Python
# Import dash
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
```
Here we try to create the homepage that contains of containers which will be filled with the things we wanted. Inside the containers there will be columns that we may fill, for example below is the column for User Country Location. The code tells that it will be given a label of "Country Location" and we can get the value by using the id "user_location_country". While the type means that the input should be a number, with minimum 1 as the country number, and maximum 195 (because there are only 195 countries in the world).<br>
```Python
# Country location
dbc.Col([
    dbc.FormGroup([
        dbc.Label("Country Location"),
        dbc.Input(id="user_location_country", placeholder="Input here ..",
                    type="number", min=1, max=195)
    ])
], md=4)
```
After creating several columns, we also would like to submit what the user have typed in the space given. The code for button is given below. <br>
```Python
dbc.Button("Submit", color="primary", block=True, id="submit-button", className="ml-auto", n_clicks=0),
```
Here is the final result of using the dash and bootstrap for the homepage. <br>
![DashScreenshot](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/home_page.PNG)<br>

### Integration and Prediction
As we don't have enough time to explore, we are unable to implement pickle to make the process faster when we process the main_dash.py. main_dash.py is the main program that contains Dash (display), Random Forest Model, and Support Vector Regression (SVR) model. This take us quite a long time to run the app as it contain the models in it. The implementation is quite the same as the one on evaluation with some added things, such as adding cluster_hotel.<br>
```Python
# Refering to a table that describe which hotel_cluster have which facilities
cluster_hotel = pd.read_csv ('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/GitHub Documentation/csv_data/hotel_cluster_csv.csv', sep=',')
```
On the code above, we place a csv file to be read and that contains facilities that a cluster have. Those facilities will also be printed in the end of the project after the prediction using both Random Forest and Support Vector Regression.<br>
The code below implies to a function that will be used to call the cluster_hotel and the facility of the hotel cluster prediction result. This code is quite simple as it only calls and determine whether a cluster have a specific facility, then return it to a variable to be printed out later on.<br>
```Python
# To determine whether the hotel have certain facility or not
def YesNo(number):
    # Hotel have the facility
    if(number==1):
        return "YES"
    # Hotel do not have the facility
    else:
        return "NO"

# Give out hotel information according to the cluster
def hotel_info (hotel_clus):
    # Print information of the hotel according to its cluster
    hotel_star="HOTEL STAR: {}. \n".format(cluster_hotel.iloc[hotel_clus-1, 1])
    hotel_wifi="WIFI: {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 2]))
    hotel_pool="POOL ACCESS: {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 3]))
    hotel_restaurant="RESTAURANT:  {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 4]))
    hotel_bar="BAR AND ALCOHOL: {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 5]))
    hotel_aircon="AIR CONDITIONER: {}. \n".format(YesNo(cluster_hotel.iloc[hotel_clus-1, 6]))
    final_info = hotel_star + hotel_wifi + hotel_pool + hotel_restaurant + hotel_bar + hotel_aircon
    return final_info
```
After some of dash code, we will find the predict hotel cluster function at the bottom which can be used to print out the final output from our prediction that includes the information of facilities. prediction1 holds the result of Random Forest prediction and hotelRFinfo will calls the function above to determine the facilities. outputRF will give us the result of prediction with the facilites. The same also implies to prediction2 which is SVR. <br>
```Python
# To get the input and prints out the final output
def predictHC(*args):
    values = list(args)
    values.pop(0)
    
    # Prediction using Random Forest
    prediction1 = rfTrain.predict([values])
    prediction1 = int(prediction1)
    outputRF = "Predicted in Random Forest Model is {}. \n".format(prediction1)
    hotelRFinfo = hotel_info(prediction1)
    outputRF = outputRF + hotelRFinfo
    
    # Prediction using Support Vector Machine
    prediction2 = svm.predict([values])
    prediction2 = int(prediction2)
    outputSVM = "Predicted in Support Vector Regression is {}. \n".format(prediction2)
    hotelSVMinfo = hotel_info(prediction2)
    outputSVM = outputSVM + hotelSVMinfo

    # Final output string
    outputFinal = outputRF + "\n" + outputSVM
    return outputFinal
```

We compile both the model that produce the result with dash that will display the program to the user. We then can see that the program work by testing it directly on  Dash. Some bit of explanation that the user can give input according to the boundaries given at the upper page, some of them is like number of continents which is 6 (Asia, North America, South America, Europe, Africa, and Australia) and etc. This comes also with an error handling if it is more than or less than the required number. If there is nothing wrong with the program, then it is ready to go. Have fun!<br>
![FinalResult](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/final_demo.gif)<br>


## References and Acknowledgements
- ProgrammingKnowledge. https://youtu.be/dX2-V2BocqQ (Install Python)
- Jake Lennon. https://youtu.be/XDdi978Xk7Y (Install Orange)
- Gourav G. Shenoy, Mangirish A. Wagle, Anwar Shaikh. https://arxiv.org/ftp/arxiv/papers/1703/1703.02915.pdf (Understanding the Data)
- Ajay Shewale. https://www.kaggle.com/ajay1216/practical-guide-on-data-preprocessing-in-python (Data Cleaning and Pre-processing)
- Harvard Data Science CS109. http://cs109.github.io/2015/pages/videos.html (Random Forest)
- Will Koehrsen. https://towardsdatascience.com/random-forest-in-python-24d0893d51c0 (Random Forest Implementation)
- Avinash Navlani. https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python (Support Vector Machine)
