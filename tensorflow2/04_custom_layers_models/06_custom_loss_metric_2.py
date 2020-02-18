"""
Handling losses and metrics that don't fit the standard signature
The overwhelming majority of losses and metrics can be computed from y_true and y_pred, where y_pred
is an output of your model.
But not all of them. For instance, a regularization loss may only require the activation of a layer
(there are no targets in this case), and this activation may not be a model output.

In such cases, you can call self.add_loss(loss_value) from inside the call method of a custom layer.
Here's a simple example that adds activity regularization
(note that activity regularization is built-in in all Keras layers --
this layer is just for the sake of providing a concrete example):
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras


class ActivityRegularizerLayer(keras.layers.Layer):
    """Gets the layer add the loss to loss in the graph and returns the layer!"""

    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs * 1e-7))
        return inputs


class MetricLoggingLayer(keras.layers.Layer):
    def call(self, inputs):
        # The 'aggregation argument defines how to aggregate the per-batch values
        # over each epoch: in this case we simply average them.
        self.add_metric(keras.backend.std(inputs), name='std_of_activation', aggregation='mean')
        return inputs  # Pass-through layer


def get_model():
    inputs = keras.Input(shape=(784,), name='digits')
    x = keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
    x = ActivityRegularizerLayer()(x)
    x = keras.layers.Dense(64, activation='relu', name='dense2')(x)
    x = ActivityRegularizerLayer()(x)
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
              metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val, y_val))
print('\nhistory dictionary:', history.history)

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss: {}, test Accuracy: {}'.format(results[0], results[1]))
predictions = model.predict((x_test[:5]))
print('shape of predictions: {}'.format(predictions.shape))

keras.backend.clear_session()


def get_model1():
    inputs = keras.Input(shape=(784,), name='digits')
    x = keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
    x = MetricLoggingLayer()(x)
    x = keras.layers.Dense(64, activation='relu', name='dense2')(x)
    x = MetricLoggingLayer()(x)
    outputs = keras.layers.Dense(10, name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model1()
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val, y_val))
print('\nhistory dictionary:', history.history)

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss: {}, test Accuracy: {}'.format(results[0], results[1]))
predictions = model.predict((x_test[:5]))
print('shape of predictions: {}'.format(predictions.shape))

"""
In the Functional API, you can also call model.add_loss(loss_tensor), 
or model.add_metric(metric_tensor, name, aggregation).

Here's a simple example:
"""
inputs = keras.Input(shape=(784,), name='digits')
x1 = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
x2 = keras.layers.Dense(64, activation='relu', name='dense_2')(x1)
outputs = keras.layers.Dense(10, name='predictions')(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

model.add_loss(tf.reduce_sum(x1) * 0.1)

model.add_metric(keras.backend.std(x1),
                 name='std_of_activation',
                 aggregation='mean')

model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x_train, y_train, batch_size=64, epochs=5)
