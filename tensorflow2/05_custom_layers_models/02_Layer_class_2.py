"""
Best Practice:
Deferring weight creation until the shape of the inputs is known.
In many cases, you may not know in advance the size of your inputs, and you would like to lazily create weights when
that value becomes known, sometimes after instantiating the layer.
In keras API, we recommend creating layer weights in the build(input_shape) method of the layer.
"""
import tensorflow as tf
from tensorflow import keras

# Previous version: We know the shape argument of input to the layer:
# We also have a quicker shortcut for adding weight to a layer using add_weight
class Linear(keras.layers.Layer):
  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units
  
  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
    self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)
   
  def call(self, inputs):
   return tf.matmul(inputs, self.w) + self.b

# At instantiation, we don't know on what inputs this is going to get called.
# The layer's weights are created dynamically the first time the layer is called.
linear_layer = Linear(32)
x = tf.ones((2, 2))
y = linear_layer(x)
