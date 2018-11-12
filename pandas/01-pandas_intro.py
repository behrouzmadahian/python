import pandas as pd
# think of Pandas as python version of Excel.
''' The data for this file is not available BUT read through it for introduction'''
df = pd.read_csv('C:/behrouz/Python/pandas/train.csv', sep=',')
print(df.head(5))
print('-'*100)
print('getting the column names:')
print(df.columns.to_list())

print('dropping unwanted columns from data frame')
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
print(df.head(5))

print('Review data types in columns:')
print(df.info())
print('Data frame shape before removing rows with NAs: ', df.shape)

#  4- removing rows with missing values:
df = df.dropna(axis=0)
#  5- removing columns with missing values:
df = df.dropna(axis=1)
print('Data frame shape after removing rows with Nas: ', df.shape)
print('Converting categorical columns into numerical')
#  how many unique categories
print(df['Sex'].unique())
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
print(df['Embarked'].unique())
df['Port'] = df['Embarked'].map({'C': 1, 'S': 2, 'Q': 3}).astype(int)
print(df.head(5))
df = df.drop(['Sex', 'Embarked'], axis=1)
print(df.head(5))

# 6- Moving columns around:
print('\n Moving columns around..\n')
cols = df.columns.tolist()
print(cols, '\n')
print('moving survived column to the beginning!')
cols = [cols[1]]+[cols[0]]+cols[2:]
print(cols)
df = df[cols]
print(df.head(5))
print('-'*200)
########################################################
# underlying numpy data!
print(df.values.shape)
