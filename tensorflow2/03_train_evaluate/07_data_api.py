"""
The tf.data API is a set of utilities in TensorFlow 2.0 for
loading and preprocessing data in a way that's fast and scalable.
You can pass a Dataset instance directly to the methods fit(), evaluate(), and predict():

Note that the Dataset is reset at the end of each epoch, so it can be reused of the next epoch.
if you want to run training only on a specific number of batches from this Dataset, you can pass the steps_per_epoch
argument, which specifies how many training steps the model should run using this Dataset
before moving on to the next epoch.

If you do this, the dataset is not reset at the end of each epoch, instead we
just keep drawing the next batches. The dataset will eventually run out of data
(unless it is an infinitely-looping dataset):
Example:
history = model.fit(train_dataset.take(100),  epochs=5, validation_data=val_dataset)

At the end of each epoch, the model will iterate over the validation Dataset and compute the validation loss and validation metrics.

If you want to run validation only on a specific number of batches from this Dataset,
you can pass the validation_steps argument, which specifies how many validation steps the model
should run with the validation Dataset before interrupting validation and moving on to the next epoch:
model.fit(train_dataset, epochs=3,
          # Only run validation using the first 10 batches of the dataset
          # using the `validation_steps` argument
          validation_data=val_dataset, validation_steps=10)

Note that the validation Dataset will be reset after each use
(so that you will always be evaluating on the same samples from epoch to epoch).

The argument validation_split (generating a holdout set from the training data) is not supported when training
from Dataset objects, since this features requires the ability to index the samples of the datasets,
 which is not possible in general with the Dataset API.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras


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

# Creating the DATA API:
# create train dDataSet instance
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# shuffle and slice the dataset
train_dataset = train_dataset.shuffle(buffer_size=2048).batch(64)

# now get the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

# since dataset already takes care of batching, we don't pass a batch_size to fit!

model = get_model()
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(train_dataset,  epochs=5, validation_data=val_dataset)
print('\nhistory dictionary:', history.history)

print('\n# Evaluate on test data')
results = model.evaluate(test_dataset)
print('test loss: {}, test Accuracy: {}'.format(results[0], results[1]))
predictions = model.predict((x_test[:5]))
print('shape of predictions: {}'.format(predictions.shape))
