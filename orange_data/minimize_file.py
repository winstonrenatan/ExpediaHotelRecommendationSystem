import pandas as pd

# read the train file to dataframe and drop rows that have no value.
df = pd.read_csv ('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/orange_data/clean_train.csv', sep=',')

# drop the item below, as it is not involved in the process later on.
# axis 1 refers to the columns.
df= df.drop('date_time', axis = 1)
df= df.drop('srch_ci', axis = 1)
df= df.drop('srch_co', axis = 1)

# .sample = Return a random sample of items from an axis of object.
# frac = Fraction of axis items to return.
# random_state = Mersenne Twister pseudo random number generator.
df = df.sample (frac=0.01, random_state=99)
# see the shape (rows and columns) of the data (exclude header).
print(df.shape)

# write the cleaned dataframe result to a new file.
export_csv = df.to_csv ('D:/TIF/TIF SEM.7/Frontier Technology/Hotel Recommendations/orange_data/clean_train_orange.csv', index = None, header=True)