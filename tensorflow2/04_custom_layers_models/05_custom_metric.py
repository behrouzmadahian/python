"""
If you need a metric that isn't part of the API, you can easily create custom metrics
by subclassing the Metric class. You will need to implement 4 methods:

1. __init__(self), in which you will create state variables for your metric.

2. update_state(self, y_true, y_pred, sample_weight=None),
which uses the targets y_true and the model predictions y_pred to update the state variables.

3. result(self), which uses the state variables to compute the final results.

4. reset_states(self), which reinitializes the state of the metric.
State update and results computation are kept separate (in update_state() and result(), respectively)
because in some cases, results computation might be very expensive, and would only be done periodically.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras


class CategoricalTruePositives(keras.metrics.Metric):

    def __init__(self, name='categorical_true_positives_rate', **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.n = self.add_weight(name='n', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """y_true is NOT one_hot"""
        y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        total = tf.cast(tf.reduce_sum(tf.ones_like(values)), 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
            total = tf.multiply(total, sample_weight)
        # update state variables
        self.true_positives.assign_add(tf.reduce_sum(values))
        self.n.assign_add(total)

    def result(self):
        return tf.divide(self.true_positives, self.n)

    def reset_states(self):
        """the state of the metric will be reset at the start of each epoch"""
        self.true_positives.assign(0.)
        self.n.assign(0.)


def get_model():
    inputs = keras.Input(shape=(784,), name='digits')
    x = keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
    x = keras.layers.Dense(64, activation='relu', name='dense2')(x)
    outputs = keras.layers.Dense(10, name='predictions')(x)
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

print(y_val.shape)
model = get_model()
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[CategoricalTruePositives()])

history = model.fit(x_train, y_train, batch_size=128, epochs=300, validation_data=(x_val, y_val))
print('\nhistory dictionary:', history.history)

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss: {}, test true positive rate: {}'.format(results[0], results[1]))
predictions = model.predict((x_test[:5]))
print('shape of predictions: {}'.format(predictions.shape))



