"""
If you don't find what you need, it's easy to extend the API by creating your own layers.

All layers subclass the Layer class and implement:

A call method, that specifies the computation done by the layer.
A build method, that creates the weights of the layer (note that this is just a style convention; y
ou could create weights in __init__ as well).
**********
If you want your custom layer to support serialization, you should also define a get_config method,
that returns the constructor arguments of the layer instance
**********

Optionally, you could also implement the class method from_config(cls, config),
which is in charge of recreating a layer instance given its config dictionary.
The default implementation of from_config is:

def from_config(cls, config):
  return cls(**config)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras


class CustomDense(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    # needed for serialization< checkpointing
    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({'units': self.units})
        return config


inputs = keras.Input(shape=(4, ))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs, outputs)
print(model.summary())
config = model.get_config()
print(config)
new_model = keras.Model.from_config(config, custom_objects={'CustomDense': CustomDense})
print(new_model.summary())


