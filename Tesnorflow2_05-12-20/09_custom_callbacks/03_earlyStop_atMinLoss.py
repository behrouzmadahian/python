"""
Early Stopping at Minimum loss!
This example showcases the creation of a Callback that stops the Keras training 
when the minimum of loss has been reached by mutating the attribute model.stop_training (boolean). 
Optionally, the user can provide an argument patience to specify how many epochs the training should 
wait before it eventually stops.

tf.keras.callbacks.EarlyStopping provides a more complete and general implementation.
Can make this on validation loss!!
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime


def get_model():
  inputs = keras.Input(shape=(784,))
  x = keras.layers.Dense(64, activation='relu')(inputs)
  x = keras.layers.Dense(64, activation='relu')(x)
  x = keras.layers.Dense(10, activation='linear')(x)
  model = keras.Model(inputs=inputs, outputs=x)
  return model


# Load example MNIST data and pre-process it
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
print(y_train[:2])


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
  # def on_train_batch_end(self, batch, logs=None):
  #   print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
  #
  # def on_test_batch_end(self, batch, logs=None):
  #   print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  def on_epoch_end(self, epoch, logs=None):
    template = 'The average loss for epoch {} is {:7.4f} and Accuracy is {:7.4f}.'
    print(template.format(epoch, logs['loss'], logs['sparse_categorical_accuracy']))


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
   """
   Stop training when the loss is at its min, i.e. the loss stops decreasing.

   Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
   """

   def __init__(self, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()

    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

   def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

   def on_epoch_end(self, epoch, logs=None):
    current = logs.get('loss')
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print('Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

   def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    
model = get_model()

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(1e-3),
              metrics=['sparse_categorical_accuracy'])

_ = model.fit(x_train, y_train,
              batch_size=64,
              steps_per_epoch=5,
              epochs=30,
              verbose=0,
              callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss(patience=10)])
