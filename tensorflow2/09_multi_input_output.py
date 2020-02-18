"""
Consider the following model, which has an image input of shape (32, 32, 3) (that's (height, width, channels))
and a timeseries input of shape (None, 10) (that's (timesteps, features)). Our model will have two outputs
computed from the combination of these inputs: a "score" (of shape (1,))
and a probability distribution over five classes (of shape (5,))
As a best practice, name output layers and do : provide dictionary for different losses and
metrics to be applied to each output

You could also chose not to compute a loss for certain outputs,
 if these outputs meant for prediction but not for training:
IN DEFINING MODEL, PUT THE OUTPUT YOU DO NOT WANT TO APPLY LOSS TO TO THE END OTHERWISE WILL RAISE ERROR!


List loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[None, keras.losses.CategoricalCrossentropy(from_logits=True)])

# Or dict loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'class_output':keras.losses.CategoricalCrossentropy(from_logits=True)})

Passing data to a multi-input or multi-output model in fit works in a similar way as specifying
a loss function in compile: you can pass lists of Numpy arrays
(with 1:1 mapping to the outputs that received a loss function)
or dicts mapping output names to Numpy arrays of training data.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras

image_input = keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = keras.Input(shape=(None, 10), name='ts_input')

x1 = keras.layers.Conv2D(3, 3)(image_input)
x1 = keras.layers.GlobalMaxPooling2D()(x1)

x2 = keras.layers.Conv1D(3, 3)(timeseries_input)
x2 = keras.layers.GlobalMaxPooling1D()(x2)

x = keras.layers.concatenate([x1, x2])

score_output = keras.layers.Dense(1, name='score_output')(x)
class_output = keras.layers.Dense(5, name='class_output')(x)

model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=[score_output, class_output])
losses = {'score_output': keras.losses.MeanSquaredError(),
          'class_output': keras.losses.CategoricalCrossentropy(from_logits=True)
         }
metrics = {'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                            keras.metrics.MeanAbsoluteError()],
           'class_output': keras.metrics.CategoricalAccuracy()
           }
loss_weights = {'score_output': 2.0, 'class_output': 1.0}
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss=losses, metrics=metrics)

# Generate dummy Numpy data
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

data_dict_x = {'img_input': img_data[:90], 'ts_input': ts_data[:90]}
data_dict_y = {'score_output': score_targets[:90], 'class_output': class_targets[:90]}

data_dict_x_v = {'img_input': img_data[90:], 'ts_input': ts_data[90:]}
data_dict_y_v = {'score_output': score_targets[90:], 'class_output': class_targets[90:]}
model.fit(data_dict_x, data_dict_y, epochs=50, batch_size=64, validation_data=(data_dict_x_v, data_dict_y_v))


# You could also chose not to compute a loss for certain outputs,
# if these outputs meant for prediction but not for training:
del model

print('-'*100)
print('Removing score loss!- use it later only for prediction!')
# PUT SCORE OUTPUT TO THE END!
model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=[class_output, score_output])
print(model.summary())
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'class_output': keras.losses.CategoricalCrossentropy(from_logits=True)},
    metrics=metrics)

model.fit(data_dict_x, data_dict_y, epochs=50, batch_size=64, validation_data=(data_dict_x_v, data_dict_y_v))

print('-'*100)
# Data API use case:
del model
keras.backend.clear_session()
print('-'*100)
print('Using data API!')

model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=[class_output, score_output])
print(model.summary())
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'class_output': keras.losses.CategoricalCrossentropy(from_logits=True)},
    metrics=metrics)
train_dataset = tf.data.Dataset.from_tensor_slices(({'img_input': img_data, 'ts_input': ts_data},
                                                    {'score_output': score_targets, 'class_output': class_targets}))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model.fit(train_dataset, epochs=3)
