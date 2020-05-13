"""
For sequences other than time series (e.g. text), it is often the case that a RNN model
can perform better if it not only processes sequence from start to end, but also backwards. 
For example, to predict the next word in a sentence, it is often useful to have the context around the word, 
not only just the words that come before it.

The output of the Bidirectional RNN will be, by default, the sum of the forward layer output and the backward
layer output. If you need a different merging behavior, e.g. concatenation, change the merge_mode parameter 
in the Bidirectional wrapper constructor.
It seems default merge_mode is concat and NOT sum!
"""
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True),
                                     input_shape=(5, 10), merge_mode='concat'))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True),
                                     input_shape=(5, 10)))
model.add(keras.layers.Dense(10))
print(model.summary())
