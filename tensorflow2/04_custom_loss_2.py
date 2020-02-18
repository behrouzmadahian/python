"""
There are two ways to provide custom losses with Keras.
The first example creates a function that accepts inputs y_true and y_pred.
If you need a loss function that takes in parameters beside y_true and y_pred,
you can subclass the tf.keras.losses.Loss class and implement the following two methods:

1. __init__(self) —Accept parameters to pass during the call of your loss function
2. call(self, y_true, y_pred) —Use the targets (y_true) and the
model predictions (y_pred) to compute the model's loss

Parameters passed into __init__() can be used during call() when calculating loss.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras


class WeightedBinaryCrossEntropy(keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, pos_weight, weight, from_logits=True,
                 reduction=keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)[:, None]
        ce = self.weight * (ce * (1 - y_true) + self.pos_weight * ce * y_true)
        return ce



def get_model():
    inputs = keras.Input(shape=(784,), name='digits')
    x = keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
    x = keras.layers.Dense(64, activation='relu', name='dense2')(x)
    outputs = keras.layers.Dense(10, name='output')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# reserve 10k samples for validation
x_val, y_val = x_train[-10000:], y_train[-10000:]
x_train, y_train = x_train[:-10000], y_train[:-10000]
# one hot encode
y_train = tf.keras.utils.to_categorical(y_train.astype(np.int32))
y_val = tf.keras.utils.to_categorical(y_val.astype(np.int32))
y_test = tf.keras.utils.to_categorical(y_test.astype(np.int32))

model = get_model()
# y is not one hot encoded so we use the sparse versions below!!!
model.compile(optimizer=keras.optimizers.Adam(),
              loss=WeightedBinaryCrossEntropy(pos_weight=1.2, weight=1, from_logits=True),
              metrics=['categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_val, y_val))
print('\nhistory dictionary:', history.history)

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss: {}, test accuracy: {}'.format(results[0], results[1]))
predictions = model.predict((x_test[:5]))
print('shape of predictions: {}'.format(predictions.shape))



