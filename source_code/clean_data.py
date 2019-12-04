# Pandas is used for data manipulation
import pandas as pd

# read the train file to dataframe and drop rows that have no value.
df = pd.read_csv ('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/first_try/raw_data/train.csv', sep=',').dropna()

# delete all rows with 0 value at is_booking
df = df[(df.is_booking != 0)]
# delete all columns below as it is presented as string and cannot be used in our algorithm
df = df.drop('is_booking', axis = 1)
df = df.drop('date_time', axis = 1)
df = df.drop('srch_ci', axis = 1)
df = df.drop('srch_co', axis = 1)

# write the cleaned dataframe result to a new file.
export_csv = df.to_csv ('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/first_try/clean_train.csv', index = None, header=True)