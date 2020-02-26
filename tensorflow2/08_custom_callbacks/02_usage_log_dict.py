"""
The logs dict contains the loss value, and all the metrics at the end of a batch or epoch. 
Example includes the loss and mean absolute error.

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

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
  def on_train_batch_end(self, batch, logs=None):
    print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
  
  def on_test_batch_end(self, batch, logs=None):
    print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  def on_epoch_end(self, epoch, logs=None):
    print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))

model = get_model()

_ = model.fit(x_train, y_train,
          batch_size=64,
          steps_per_epoch=5,
          epochs=3,
          verbose=0,
          callbacks=[LossAndErrorPrintingCallback()])
  
 

    
_ = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=20,
          callbacks=[LossAndErrorPrintingCallback()])

