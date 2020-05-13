"""
restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored quantity.
If False, the model weights obtained at the last step of training are used.
NOTE: it seems:
cannot use tf.keras.callbacks.ReduceLROnPlateau and lr_schedule at the same time
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

class_output = keras.layers.Dense(5, name='class_output')(x)

model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=class_output)

# Generate dummy Numpy data
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
class_targets = np.random.random_sample(size=(100, 5))

data_dict_x = {'img_input': img_data[:90], 'ts_input': ts_data[:90]}
data_y = class_targets[:90]

data_dict_x_v = {'img_input': img_data[90:], 'ts_input': ts_data[90:]}
data_y_v = class_targets[90:]

# "no longer improving" being defined as min delta "no better than 1e-5 less"
# patience: "no longer improving" being further defined as "for at least 20 epochs"
checkpoint_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/chkpoint_files'

checkpoint_path = checkpoint_folder + "/model.ckpt"
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=1e-5,
                                                        patience=20,
                                                        verbose=2,
                                                        mode='min',  # if val loss stops decreasing stop...
                                                        restore_best_weights=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(monitor='val_categorical_accuracy',
                                                 filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 mode='max',
                                                 save_best_only=True
                                                 )
"""
new_lr = lr * factor
cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
"""
lr_decay_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy',
                                                         factor=0.5, patience=10, verbose=0,
                                                         mode='max', min_delta=1e-5, cooldown=0,
                                                         min_lr=1e-6)
initial_learning_rate = 1e-3
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                          decay_steps=100000,
                                                          decay_rate=0.96,
                                                          staircase=True)
checkpoint_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/chkpoint_files'

tensorboard_cbk = keras.callbacks.TensorBoard(log_dir=checkpoint_folder,
                                              histogram_freq=0,  # How often to log histogram visualizations
                                              embeddings_freq=0,  # How often to log embedding visualizations
                                              update_freq='epoch',  # How often to write logs (default: once per epoch)
                                              write_graph=True)

loss = keras.losses.CategoricalCrossentropy(from_logits=True)
# feed metrics as a list!
metrics = [keras.metrics.CategoricalAccuracy()]
# Learning rate schedule:
# optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
optimizer = keras.optimizers.RMSprop(learning_rate=initial_learning_rate)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
callbacks = [early_stopping_callback, cp_callback, lr_decay_callback, tensorboard_cbk]
model.fit(data_dict_x, data_y, epochs=500, batch_size=64,
          validation_data=(data_dict_x_v, data_y_v), callbacks=callbacks)

