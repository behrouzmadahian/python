"""
In the functional API, models are created by specifying their inputs and outputs in a graph of layers.
That means that a single graph of layers can be used to generate multiple models.

In the example below, we use the same stack of layers to instantiate two models:
- an encoder model : that turns image inputs into 16-dimensional vectors,
- and an end-to-end autoencoder model for training.

You can treat any model as if it were a layer, by calling it on an Input or on the output of another layer.
Note that by calling a model you aren't just reusing the architecture of the model, you're also reusing its weights.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.keras.backend.clear_session()

kernel_init = keras.initializers.GlorotUniform()
bias_init = keras.initializers.constant(0.1)

encoder_input = keras.Input(shape=(28, 28, 1), name='img')

x = keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='valid',
                        dilation_rate=(1, 1), activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(encoder_input)

x = keras.layers.Conv2D(32, 3, activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)

x = keras.layers.MaxPooling2D(3)(x)

x = keras.layers.Conv2D(32, 3, activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)

x = keras.layers.Conv2D(16, 3, activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init)(x)

encoder_output = keras.layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
print(encoder.summary())

x = keras.layers.Reshape((4, 4, 1))(encoder_output)
x = keras.layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = keras.layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = keras.layers.UpSampling2D(3)(x)
x = keras.layers.Conv2DTranspose(16, 3, activation='relu')(x)

decoder_output = keras.layers.Conv2DTranspose(1, 3, activation='relu')(x)

auto_encoder = keras.Model(encoder_input, decoder_output, name='auto_encoder')
print('\n\nFull Auto encoder model summary: \n')

print(auto_encoder.summary())
auto_encoder.compile(loss=keras.losses.mse,
                     optimizer=keras.optimizers.RMSprop(),
                     metrics=['mse'])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0
print(x_train.shape)
auto_encoder.fit(x_train, x_train, epochs=3, batch_size=128)




