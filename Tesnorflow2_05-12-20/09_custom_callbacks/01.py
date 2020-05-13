"""
You can pass a list of callbacks (as the keyword argument callbacks) to any of 
tf.keras.Model.fit(), tf.keras.Model.evaluate(), and tf.keras.Model.predict() methods. 
The methods of the callbacks will then be called at different stages of training/evaluating/inference.
Define a simple custom callback to track the start and end of every batch of data.
During those calls, it prints the index of the current batch.

Model methods that take callbacks:
Users can supply a list of callbacks to following tf.keras.Model methods
fit(), fit_generator()

evaluate(), evaluate_generator()
- Evaluates the model for given data or data generator. Outputs the loss and metric values from the evaluation.

predict(), predict_generator()
Generates output predictions for the input data or data generator.

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
  
  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    
  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_predict_batch_begin(self, batch, logs=None):
      print('Predicting: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_predict_batch_end(self, batch, logs=None):
      print('Predicting: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
  
  # you can have the _end versions as well!
  def on_train_begin(self, logs=None):
      """ Called at the begining of training"""
      print('Training Started at {}'.format(datetime.datetime.now().time()))
  
  def on_test_begin(self, logs=None):
      """ Called at the begining of evaluation"""
      print('Evaluation Started:  at {}'.format(datetime.datetime.now().time()))
      
  def on_predict_begin(self, logs=None):
      """ Called at the begining of prediction"""
      print('Predicting: Started at {}'.format(datetime.datetime.now().time()))
  
  def on_epoch_begin(self, epoch, logs=None):
      """ Called at the begining of each epoch!"""
      print('Epoch {} Starts at {}'.format(epoch, datetime.datetime.now().time()))
  
  def on_epoch_end(self, epoch, logs=None):
      """ Called at the End of each epoch!"""
      print('Epoch {} Starts at {}'.format(epoch, datetime.datetime.now().time()))


history = model.fit(x_train, y_train, batch_size=64, epochs=1, steps_per_epoch=5, verbose=0, callbacks=[MyCustomCallback()])

eval_hist = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=5, callbacks=[MyCustomCallback()])
