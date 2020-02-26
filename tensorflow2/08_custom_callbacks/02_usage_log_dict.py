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

class MyCustomCallback(keras.callbacks.Callback):
  
  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
  
  def on train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    
  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_predict_batch_begin(self, logs=None):
      print('Predicting: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_predict_batch_end(self, logs=None):
      print('Predicting: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
  
  # you can have the _end versions as well!
  def on_train_begin(self, logs=None):
      """ Called at the begining of training"""
      print('Training Started at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_test_begin(self, logs=None):
      """ Called at the begining of evaluation"""
      print('Evaluation Started:  at {}'.format(batch, datetime.datetime.now().time()))
      
  def on_predict_begin(self, logs=None):
      """ Called at the begining of prediction"""
      print('Predicting: Started at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_epoch_begin(self, epoch, logs=None):
      """ Called at the begining of each epoch!"""
      print('Epoch {} Starts at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_epoch_end(self, epoch, logs=None):
      """ Called at the End of each epoch!"""
      print('Epoch {} Starts at {}'.format(batch, datetime.datetime.now().time()))

    


history = model.fit(x_train, y_train, batch_size=64, epochs=1, steps_per_epoch=5, verbose=0, callbacks=[MyCustomCallback()])

eval_hist = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=5, callbacks=[MyCustomCallback()])


                    
