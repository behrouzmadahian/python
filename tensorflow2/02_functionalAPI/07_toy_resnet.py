"""
In addition to models with multiple inputs and outputs, the Functional API makes it easy to manipulate
non-linear connectivity topologies, that is to say, models where layers are not connected sequentially.
This also cannot be handled with the Sequential API (as the name indicates).
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

checkpoint_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/chkpoint_files'

tf.keras.backend.clear_session()
kernel_init = keras.initializers.GlorotUniform()
bias_init = keras.initializers.constant(0.1)

inputs = keras.Input(shape=(32, 32, 3), name='img')
x = keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=(1, 1), activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(inputs)

x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=(1, 1), activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)

block_1_output = keras.layers.MaxPooling2D(3)(x)
#
x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=(1, 1), activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(block_1_output)

x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=(1, 1), activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)

block_2_output = keras.layers.add([x, block_1_output])
#
x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=(1, 1), activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(block_2_output)

x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=(1, 1), activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)

block_3_output = keras.layers.add([x, block_2_output])
#

x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=(1, 1), activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(block_3_output)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x, training=True)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name='toy_resnet')
print(model.summary())


keras.utils.plot_model(model, checkpoint_folder+'/mini_resnet.png', show_shapes=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print('Shape of y_train: {}, y_test: {}, after turning into one hot encoding.'.format(y_train.shape, y_test.shape))

model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss = keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['acc'])

model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
