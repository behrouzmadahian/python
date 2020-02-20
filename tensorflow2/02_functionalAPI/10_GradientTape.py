"""
You can write you own training and evaluation loops: Using GradientTape
Calling a model inside a GradientTape scope enables you to retrieve the gradients of 
the trainable weights og the layer with respect to a loss value.
we can use these gradients to update these variables which we can retrieve using model.trainable_weights
"""
from tensorflow import keras
import tensorflow as tf


def create_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = keras.layers.Dense(64, activation'relu', name='dense_1')(inputs)
  x = keras.layers.Dense(64, activation'relu', name='dense_2')(x)
  outputs = keras.layers.Dense(10, name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model


model = create_model()
optimizer = keras.optimizer.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Training loop:

                                

  
