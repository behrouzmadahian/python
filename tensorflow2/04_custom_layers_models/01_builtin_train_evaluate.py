"""
When passing data to the built-in training loops of a model, you should either use Numpy arrays
(if your data is small and fits in memory) or tf.data Dataset objects.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras

inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
x = keras.layers.Dense(64, activation='relu', name='dense2')(x)
outputs = keras.layers.Dense(10, activation='relu', name='output')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# reserve 10k samples for validation
x_val, y_val = x_train[-10000:], y_train[-10000:]
x_train, y_train = x_train[:-10000], y_train[:-10000]

# y is not one hot encoded so we use the sparse versions below!!!
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_val, y_val))
print('\nhistory dictionary:', history.history)

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss: {}, test accuracy: {}'.format(results[0], results[1]))
predictions = model.predict((x_test[:5]))
print('shape of predictions: {}'.format(predictions.shape))

