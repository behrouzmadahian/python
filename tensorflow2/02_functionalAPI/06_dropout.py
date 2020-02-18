"""
Need Functional API model.
1. train the model in training mode and save  model weights
2. create the model, set the training flag to false
3. load the model weights in it and do predictions
4. save this model in complete .h5 or SavedModel format
5. now you have the model saved in inference mode without the need for access to original code!!

we could save model weights, then create the model in test mode and load the weights
- if we wanted to only load the weights
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.clear_session()


def create_model(is_train=None, drop_rate=0.1):
    """
    :param is_train: If true, training mode. default None: disables dropout
    :return: model
    """
    inputs = keras.Input(shape=(784,), name='image_input')
    print('Shape and data type of input layer: {}, {}'.format(inputs.shape, inputs.dtype))
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dropout(drop_rate)(x, training=is_train)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(drop_rate)(x, training=is_train)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    return model


# at this point we can create a Model by specifying inputs and outputs in the graph of layers
model = create_model(is_train=True, drop_rate=0.3)
print(model.summary())

checkpoint_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/chkpoint_files'
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0


checkpoint_path = checkpoint_folder + "/model.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 mode='max',
                                                 save_best_only=True
                                                 )

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128, epochs=5,
                    validation_split=0.2, verbose=2, callbacks=[cp_callback])

model.load_weights(checkpoint_path)
test_eval = model.evaluate(x_test, y_test, verbose=2)
print('\n\nTest loss and accuracy: - Train mode: {}\n\n'.format(test_eval))

full_model_name = checkpoint_folder + '/trainMode_func_API_SavedModel'

del model
tf.keras.backend.clear_session()

model = create_model(is_train=None)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
model.load_weights(checkpoint_path)
test_eval = model.evaluate(x_test, y_test, verbose=2)
print('\n\nTest loss and accuracy: - Test mode before saving to file : {}'.format(test_eval))
model.save(full_model_name)


del model
tf.keras.backend.clear_session()

model = keras.models.load_model(full_model_name)
print(model.summary())

test_eval = model.evaluate(x_test, y_test, verbose=2)
print('\n\nTest loss and accuracy: - Test mode After saving Full model to file in inference mode : {}'.format(test_eval))
