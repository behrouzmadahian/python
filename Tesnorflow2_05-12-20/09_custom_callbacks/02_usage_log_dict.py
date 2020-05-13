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
  x = keras.layers.Dense(64, activation='relu')(inputs)
  x = keras.layers.Dense(64, activation='relu')(x)
  x = keras.layers.Dense(10, activation='linear')(x)
  model = keras.Model(inputs=inputs, outputs=x)
  return model


model = get_model()
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.1), loss='mse', metrics=['mse'])

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


model = get_model()
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(1e-3),
              metrics=['sparse_categorical_accuracy'])
print(model.summary())
_ = model.fit(x_train, y_train,
              batch_size=64,
              #steps_per_epoch=5,
              epochs=30,
              verbose=0,
              callbacks=[LossAndErrorPrintingCallback()])
print('*'*100)
_ = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=20,
                   callbacks=[LossAndErrorPrintingCallback()])

