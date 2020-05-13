"""
Using sample weighting and class weighting
Besides input data and target data, it is possible to pass sample weights or class weights to a model when using fit:

- When training from Numpy data:
        via the sample_weight and class_weight arguments.
- When training from Datasets:
        by having the Dataset return a tuple (input_batch, target_batch, sample_weight_batch) .

A "sample weights" array is an array of numbers that specify how much weight each sample in a batch should have
in computing the total loss.

A "class weights" dict is a more specific instance of the same concept: it maps class indices
to the sample weight that should be used for samples belonging to this class.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Set weight "2" for class "5",
# making this class 2x more important
class_weights = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1.,
                 5: 2.,
                 6: 1., 7: 1., 8: 1., 9: 1.}


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

sample_weight = np.ones(shape=(len(y_train),))
print('sample weight shape: {}'.format(sample_weight.shape))
# Does the same thing as class weight!!!! but more general to non classification problems
sample_weight[y_train == 5] = 1.5
model = get_model()
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])


history = model.fit(x_train, y_train,  epochs=5, validation_data=(x_val, y_val),
                    sample_weight=sample_weight, class_weight=class_weights)
print('\nhistory dictionary:', history.history)

print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test)
print('test loss: {}, test Accuracy: {}'.format(results[0], results[1]))
predictions = model.predict((x_test[:5]))
print('shape of predictions: {}'.format(predictions.shape))

# Matching example using datset API:
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.

# Create a Dataset that includes sample weights
# (3rd element in the return tuple).
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))

# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
# now get the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

keras.backend.clear_session()
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
