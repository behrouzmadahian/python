"""
Some layers, in particular BatchNormalization layer and the Dropout layer, have different behaviors during
training and inference. For such layers, it is standard practice to expose a training(boolean) argument
in the call method By exposing this argument in call, you enable the built-in training and evaluation loops
to correctly use the layer in training and inference
"""

import tensorflow as tf
from tensorflow import keras


class CustomDropout(keras.layers.Layer):
  def __init__(self, rate, **kwargs):
    super(CustomDropout, self).__init__(**kwargs)
    self.rate = rate
  
  def call(self, inputs, training=None):
    if training:
        return tf.nn.dropout(inputs, rate=self.rate)
    return inputs

