"""
The Functional API is a way to create models that ARE more flexible than Sequential: it can handle:
- models with non-linear topology
- models with shared layers,
- models with multiple inputs or outputs.
Functional API provides set of tools for building graph of layers

- To build models with Functional API, you start by creating an input node!
Note:
    in input layer, the batch size is ALWAYS omitted from shape argument

- Create a new node in the graph of layers by calling a layer on the inputs object
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

tf.keras.backend.clear_session()

inputs = keras.Input(shape=(784,), name='image_input')
print('Shape and data type of input layer: {}, {}'.format(inputs.shape, inputs.dtype))
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.9)(x, training=True)
outputs = keras.layers.Dense(10)(x)

# At this point we can create a Model by specifying inputs and outputs in the graph of layers
model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
print(model.summary())

# plot model graph:
checkpoint_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/chkpoint_files'

keras.utils.plot_model(model, checkpoint_folder+'/model_graph.png', show_shapes=True)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=2,
                    validation_split=0.2, verbose=2)

print(type(x_train[:1]))
x_random = np.random.random((1, 784))
model_out_train = model(x_random, training=False)
print(model_out_train.shape)
model_out_inference = model(x_random, training=False)
model_out_pred_func = model.predict(x_random)
print(model_out_pred_func.shape)
print('predictions: train: \n', model_out_train)
print('-'*100)
print('Predictions Inference: \n {}'.format(model_out_inference))
print('-'*100)

print("predictions using predict function \n {}".format(model_out_pred_func))

"""
the History.history attribute is a record of training loss values and metrics values at successive epochs,
as well as validation loss values and validation metrics values (if applicable)
"""
print('History of train returned by model.fit: \n')
print(history.history)
test_scores = model.evaluate(x_test, y_test, verbose=0)
print('\n\n Test loss: {}, Test Accuracy: {}'.format(test_scores[0], test_scores[1]))
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.save(checkpoint_folder + '/func_API_SavedModel')
del model
model = keras.models.load_model(checkpoint_folder + '/func_API_SavedModel')
print(model.summary())
