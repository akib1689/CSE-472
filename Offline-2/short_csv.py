# file to create short csv files for testing

import pandas as pd

# Take input file name from user
INPUT_FILE = input('Enter input file name: ')

df = pd.read_csv(INPUT_FILE)

# take the number of rows to be extracted from user
n = int(input('Enter number of rows to be extracted: '))

# create a new dataframe with the first n rows
df1 = df.head(n)

# export the dataframe to csv
df1.to_csv('short.csv', index=False)
