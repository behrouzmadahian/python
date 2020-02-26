"""
Early Stopping at Minimum loss!
First example showcases the creation of a Callback that stops the Keras training 
when the minimum of loss has been reached by mutating the attribute model.stop_training (boolean). 
Optionally, the user can provide an argument patience to specify how many epochs the training should 
wait before it eventually stops.

tf.keras.callbacks.EarlyStopping provides a more complete and general implementation.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import datetime

def get_model():
  inputs = keras.Input(shape=(784,))
  x = keras.layers.Dense(1, activation='linear')(inputs)
  model = keras.Model(inputs=inputs, outputs=x)
  return model

model = get_model()
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.1), loss='mse', metrics=['mse'])

# Load example MNIST data and pre-process it
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
