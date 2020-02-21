"""
If you assign a Layer instance as attribute of another layer, 
the outer layer will start tracking the weights og the inner layer
We recommend creating such sublayers in the __init__ method
(since the sublayers will typically have a build method,
they will be built when the outer layer gets built
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

class MLPBlock(layers.Layer):
  def __init__(self):
    super(MLPBlock, self).__init__()
    self.linear1 = Linear(32)
    self.linear2 = Linear(32)
    self.linear3 = Linear(1)
  
  def call(self, inputs):
    x = self.linear1(inputs)
    x = tf.nn.relu(x)
    x = self.linear2(x)
    x = tf.nn.relu(x)
    return self.linear3(x)
  
mlp = MLPBlock()
# the first call to mlp will create the weights
y = mlp(tf.ones(shape=(3, 64)))
print("weights: {}".format(len(mlp.weights)))
print('trainable weights: {}'.format(mlp.trainable_weights))
