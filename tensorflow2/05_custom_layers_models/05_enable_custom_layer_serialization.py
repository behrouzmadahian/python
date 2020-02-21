"""
If you need your custom layer to be serializable as part of Functional model, 
you can optionally implement the get_config() method
"""
import tensorflow as tf
from tensorflow import keras


class Linear(keras.layers.Layer):
  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units
  
  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units), 
                             initialier='random_normal', trainable=True)
    self.b = self.add_weight(shape=(self.units,), 
                             initializer='random_normal', trainable=True)
    
  def call(self):
    return tf.matmul(inputs, self.w) + self.b
  
  def get_config(self):
    config = {'units': self.units}
    return config

  
# Now you can ecreate the layer from its config:
layer = Linear(64)
config = layer.get_config()
print(config)

new_layer = Linear.from_config(config)

"""
Note that the __init__ method of the base Layer class takes some keyword arguments, 
in particular a < name >  and a < dtype >.
It is good practice to pass these arguments to the parent class __init__ 
and to inclue them in the layer config.
"""
class Linear(keras.layers.Layer):
  def __init__(self, units=32, **kwargs):
    super(Linear, self).__init__(**kwargs)
    self.units = units
  
  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units), 
                             initialier='random_normal', trainable=True)
    self.b = self.add_weight(shape=(self.units,), 
                             initializer='random_normal', trainable=True)
    
  def call(self):
    return tf.matmul(inputs, self.w) + self.b
  
  def get_config(self):
    config = super(Linear, self).get_config()
    config.update({'units': self.units})
    return config
  

layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)

"""
If you need more flexibility when deserializing the layer from its config, you can also override the 
from_config class method.
This is the base implementation of from_config:

def from_config(cls, config):
  return cls(**config)
"""
  
                             
