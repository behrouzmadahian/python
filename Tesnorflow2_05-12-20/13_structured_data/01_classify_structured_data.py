"""
This tutorial demonstrates how to classify structured data (e.g. tabular data in a CSV).
We will use Keras to define the model, and feature columns as a bridge to map from columns
in a CSV to features used to train the model. This tutorial contains complete code to:

- Load a CSV file using Pandas.
- Build an input pipeline to batch and shuffle the rows using tf.data.
- Map from columns in the CSV to features used to train the model using feature columns.
- Build, train, and evaluate a model using Keras.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
print('Head of data set: ')
print(dataframe.head())
print(dataframe.columns)

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print("Train Examples :{}".format(train.shape))
print("validation Examples :{}".format(val.shape))
print("Test Examples :{}".format(test.shape))

# Create input pipeline using tf.data
"""
Next, we will wrap the dataframes with tf.data. This will enable us to use feature columns as a 
bridge to map from the columns in the Pandas dataframe to features used to train the model. 
If we were working with a very large CSV file (so large that it does not fit into memory), 
we would use tf.data to read it from disk directly. That is not covered in this tutorial.
"""
print(dict(test).keys())
print(dict(test)[list(dict(test).keys())[0]][:10])


# A utility function to create tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

"""
We can see that the dataset returns a dictionary of column names (from the dataframe) 
that map to column values from rows in the dataframe.
"""
# Understanding input pipeline:
for feature_batch, label_batch in train_ds.take(1):
    print('Every Feature: \n{}'.format(list(feature_batch.keys())))
    print('A batch of ages: \n{}'.format(feature_batch['age']))
    print('A batch of targets:\n {}'.format(label_batch))


"""
TensorFlow provides many types of feature columns. In this section, we will create several 
types of feature columns, and demonstrate how they transform a column from the dataframe.
"""
# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]


# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

"""
Numeric columns:
The output of a feature column becomes the input to the model 
(using the demo function defined above, we will be able to 
 see exactly how each column from the dataframe is transformed).
A numeric column is the simplest type of column. It is used to represent real valued features. 
When using this column, your model will receive the column value from the dataframe unchanged.
"""

age = tf.feature_column.numeric_column('age')
print(age)
age_feat_column = demo(age)
print(age_feat_column)

# Bucketized Columns:
"""
Often, you don't want to feed a number directly into the model, but instead split its value into different
 categories based on numerical ranges. Consider raw data that represents a person's age. 
 Instead of representing age as a numeric column, we could split the age into several buckets 
 using a bucketized column. Notice the one-hot values below describe which age range each row matches.
"""
print('Bucketized AGE column: ONE hot!')
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 35, 40, 45, 50, 55, 60, 65])
print(demo(age_buckets))

# Categorical Columns:
"""
In this dataset, thal is represented as a string (e.g. 'fixed', 'normal', or 'reversible'). 
We cannot feed strings directly to a model. Instead, we must first map them to numeric values. 
The categorical vocabulary columns provide a way to represent strings as a one-hot vector 
(much like you have seen above with age buckets). 
The vocabulary can be passed as a list using categorical_column_with_vocabulary_list, 
or loaded from a file using categorical_column_with_vocabulary_file.
"""
print('Categorical columns: ')
thal = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
print(thal)
thal_one_hot = tf.feature_column.indicator_column(thal)
print(demo(thal_one_hot))

"""
In a more complex dataset, many columns would be categorical (e.g. strings).
 Feature columns are most valuable when working with categorical data. 
 Although there is only one categorical column in this dataset, we will use 
 it to demonstrate several important types of feature columns that you could use when working with other datasets.
"""
# Embedding columns
"""
Suppose instead of having just a few possible strings, we have thousands (or more) values per category. 
For a number of reasons, as the number of categories grow large, it becomes infeasible to train a neural 
network using one-hot encodings. We can use an embedding column to overcome this limitation. 
Instead of representing the data as a one-hot vector of many dimensions, an embedding column represents 
that data as a lower-dimensional, dense vector in which each cell can contain any number, not just 0 or 1. 
The size of the embedding (8, in the example below) is a parameter that must be tuned.
"""
# Notice the input to the embedding column is the categorical column
# we previously created
print('Turning Categorical column to embedding column of size 8!')
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
print(demo(thal_embedding))

# Crossed feature columns
"""
Combining features into a single feature, better known as feature crosses, enables a model to learn 
separate weights for each combination of features. 
Here, we will create a new feature that is the cross of age and thal. 
Note that crossed_column does not build the full table of all possible combinations 
(which could be very large). Instead, it is backed by a hashed_column, so you can choose how large the table is.
"""
print('Cross columns: one categorical one continuous!')
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
print(demo(tf.feature_column.indicator_column(crossed_feature)))

"""
We have seen how to use several types of feature columns. Now we will use them to train a model. 
The goal of this tutorial is to show you the complete code (e.g. mechanics) 
needed to work with feature columns. We have selected a few columns to train our model below arbitrarily.
"""

feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(tf.feature_column.numeric_column(header))

# bucketized cols
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = tf.feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

# Create a feature layer:
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Now that we have defined our feature columns, we will use a DenseFeatures layer to input them to our Keras model.
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


model = tf.keras.Sequential([
  feature_layer,
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=5)
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
