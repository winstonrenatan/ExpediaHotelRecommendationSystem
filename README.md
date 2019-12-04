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
Guide on how to install: https://youtu.be/XDdi978Xk7Y <br>
[Python 3.7](https://www.python.org/downloads/)<br>
Guide on how to download: https://youtu.be/dX2-V2BocqQ <br>
Besides the Python itself, we also need some package to import such as below:<br>
[Pandas](https://pandas.pydata.org/)<br>
[NumPy](https://numpy.org/)<br>
[scikit-learn](https://scikit-learn.org/stable/)<br>
Or we can just simply download the whole thing in [Anaconda](https://www.anaconda.com/distribution/).<br>
Don't forget to also to install [Dash](https://dash.plot.ly/installation) for the display.<br>

## Understanding the Data
### test.csv
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
This data contains information about the hotel reviews made by users and extracted as a features.<br>

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
Here what we wanted to do is to clean the data so that it is more compact and clear. First of all, we wanted to clean the data that have no value in one of the column. For example as the picture below, that did not have the value for orig_destination_distance.<br>
![NoValueOnOrigDestinationDistance](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/orig_destination_distance_value_nan.png)<br>
The next thing is that, we would like to focus just on the data where is_booking value is 1. Why? it is because when is_booking value equals to 0, people just seeing through (window shopping) but not booking or doing the transaction.<br>
![IsBookingValue](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/is_booking_value_one.png)<br>
Besides, we would also like to cut the column for date_time, srch_ci, and srch_co. As the algorithm we provide cannot work with strings and dates.<br>
![DateValue](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/date_value_string.png)<br>

### Orange Evaluation
Here we would like to see which algorithm or model that works with our data. We will evaluate using the data that have been cleaned and also once randomize it just using the 1% of the data (19855 data) so that Orange can run smooth on my machine. To cut the data for Orange, we can use the minimize_file.py. The result below shows the result of using Orange, that our data works with Random Forest and Support Vector Machine, thus that is going to be implemented.<br>
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
Random Forest itself builds upon the idea of bagging which reduces overfiting. This model have two parameters which are, the number of trees and the number of features. The error will depends on correlation between trees and strength of each single tree. This method is also easy to parallelize, that makes it good for use. Here using the rf_evaluation.py for the evaluation we get the result as below.<br>

|Evaluation        |Value                     |
|------------------|--------------------------|
|MAE               |28.2                      |
|MSE               |1430.1012903022558        |
|RMSE              |37.81668005394254         |
|R<sup>2</sup>     |-0.6820501555223537       |

Support Vector Regression can both be applied to solve problems in classification and regression. It also able to endure with multiple variables. There are many methods (kernel we say) to work with such as Linear, Polynomial, Radial Bias Function, and much more. This model gives out good accuracy and use less memory. On the downside, SVR comes with long training time for large dataset. Here using the svr_evaluation.py for the evaluation we get the result as below.<br>

|Evaluation        |Value                     |
|------------------|--------------------------|
|MAE               |24.55                     |
|MSE               |845.3880810817358         |
|RMSE              |29.307555813878275        |

### Dash Development
At the same time with Model Train and Test, we also would like to display our result not just in the terminal so we develop the front-end with Python Dash. This development also uses bootstrap. <br>
![DashScreenshot](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/home_page.PNG)<br>

### Integration and Prediction
We compile both the model that produce the result with dash that will display the program to the user. We then can see that the program work by testing it directly on  Dash. If there is nothing wrong with the program, then it is ready to go. Have fun!<br>
Some bit of explanation that the user can give input according to the boundaries given at the upper page, some of them is like number of continents which is 6 (Asia, North America, South America, Europe, Africa, and Australia) and etc. This comes also with an error handling if it is more than or less than the required number.<br>
![FinalResult](https://github.com/winstonrenatan/ExpediaHotelRecommendationSystem/blob/master/visual_documentation/final_demo.gif)<br>


## References and Acknowledgements
- ProgrammingKnowledge. https://youtu.be/dX2-V2BocqQ (Install Python)
- Jake Lennon. https://youtu.be/XDdi978Xk7Y (Install Orange)
- Gourav G. Shenoy, Mangirish A. Wagle, Anwar Shaikh. https://arxiv.org/ftp/arxiv/papers/1703/1703.02915.pdf (Understanding the Data)
- Ajay Shewale. https://www.kaggle.com/ajay1216/practical-guide-on-data-preprocessing-in-python (Data Cleaning and Pre-processing)
- Harvard Data Science CS109. http://cs109.github.io/2015/pages/videos.html (Random Forest)
- Will Koehrsen. https://towardsdatascience.com/random-forest-in-python-24d0893d51c0 (Random Forest Implementation)
- Avinash Navlani. https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python (Support Vector Machine)
