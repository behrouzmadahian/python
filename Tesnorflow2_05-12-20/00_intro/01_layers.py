import tensorflow as tf
from tensorflow import keras

layer = keras.layers.Dense(10)
# A layer will be initialized the first time it is called!
print('Layer variables before initialization(Layer being called):')
print(layer.variables)
x = layer(tf.zeros([20, 3]))
# Getting all the variables a layer has:\
print('Layer variables AFTER initialization(Layer being called):')

print(layer.variables)

print('The kernel weights:')
print(layer.kernel)
print('bias weights:')
print(layer.bias)

