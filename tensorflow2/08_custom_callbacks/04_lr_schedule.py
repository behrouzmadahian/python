"""
Learning rate schedule
see callbacks.LearningRateScheduler and keras.optimizers.schedules for more general implementations.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import datetime

class LearningRateScheduler(tf.keras.callbacks.Callback):
  """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

  def __init__(self, schedule):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    # Call schedule function to get the scheduled learning rate.
    scheduled_lr = self.schedule(epoch, lr)
    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))

    
class LossAndErrorPrintingCallback(keras.callbacks.Callback):
  def on_train_batch_end(self, batch, logs=None):
    print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
  
  def on_test_batch_end(self, batch, logs=None):
    print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  def on_epoch_end(self, epoch, logs=None):
    print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))

    
LR_SCHEDULE = [
                # (epoch to start, learning rate) tuples
                (3, 0.05), (6, 0.01), (9, 0.005), (12, 0.001)
              ]


def lr_schedule(epoch, lr):
  """Helper function to retrieve the scheduled learning rate based on epoch."""
  if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
    return lr
  for i in range(len(LR_SCHEDULE)):
    if epoch == LR_SCHEDULE[i][0]:
      return LR_SCHEDULE[i][1]
  return lr


def get_model():
  inputs = keras.Input(shape=(784,))
  x = keras.layers.Dense(1, activation='linear')(inputs)
  model = keras.Model(inputs=inputs, outputs=x)
  return model


# Load example MNIST data and pre-process it
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model = get_model()
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.1), loss='mse', metrics=['mse'])

_ = model.fit(x_train, y_train,
          batch_size=64,
          steps_per_epoch=5,
          epochs=15,
          verbose=0,
          callbacks=[LossAndErrorPrintingCallback(), LearningRateScheduler(lr_schedule)])
