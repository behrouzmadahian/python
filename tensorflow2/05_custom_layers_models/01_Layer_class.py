"""
The main data structure you will work with is the Layer. a layer encapsulates both a state(the layer's weights)
and a transformation from inputs to outputs( a "call", the layer's forward pass).
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.clear_session()

class Linear(keras.layers.Layer):
  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value= w_init(shape=(input_dim, units), dtype='tfloat32'), trainable=True)
    
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype='float32'), trainable=True)
    
  def call(self, inputs):
   return tf.matmul(inputs, self.w) + self.b

x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
# Note that the weights w and b are automatically tracked by the layer upon being set as layer attributes:
assert linear_layer.weights == [linear_layer.w, linear_layer.b]

# We also have a quicker shortcut for adding weight to a layer using add_weight
class Linear(keras.layers.Layer):
  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal', trainable=True)
    self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True)
   
  def call(self, inputs):
   return tf.matmul(inputs, self.w) + self.b

x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)

"""
Layers can have non-trainable weights:
Besides trainable weights, you can add non-trainable weights to a layer as well.
Such weights are meant not to be taken into account during back propagation. When you are training the layer.
The new variable becomes part of the layer.weights, but it gets categorized as a non-trainable weight
"""
class ComputeSum(layers.Layer):
  def __init__(self, input_dim):
    super(Compute_sum, self).__init__()
    self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)
  
  def call(self, inputs):
    self.total.assign_add(tf.reduce_sum(inputs, axis=0))
    return total
  
x = tf.ones((2,2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
print('Calling mysum object again, will add the internal state!')
y = mysum(x)
print("weights: {}".format(len(my_sum.weights)))
print('Non-trainable weights: {}'.format(my_sum.non_trainable_weights))
print('Non trainable variable is not included in list of trainable variables...')
print('trainable weights: {}'.format(my_sum.trainable_weights))

      

      
      
print(y.numpy())

                                             
