"""
When we pad the data, we need to inform the model that some part of the data is padded
and must be ignored.
There are 3 ways to introduce input masks in keras models:
-  Add keras.layers.Masking layer
-  Configure keras.layers.Embedding  layer with mask_zero=True
-  Pass a mask argument manually when calling layers that support this argument(e.g RNN layers)

Mask generating layers: Embedding, Masking:
under the hood, these layers will create a mask tensor (2D tensor with shape (Batch, sequence_length))
and attach it to the tensor output returned by the Masking or Embedding layer!


# If the masked values are in the middle of sequence, need to make sure
we correctly mask it before sending to LSTM:
the outputs of masked steps will be skipped in LSTM calculations!!
Note that I explicitly multiply the output of LSTM to masked tensor.
This is due to the fact that the output of LSTM is not masked!!
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras


raw_inputs = [[83, 91, 1, 645, 1253, 927],
              [73, 8, 3215, 55, 927],
              [711, 632, 71]]
padded_input = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, maxlen=None,
                                                             dtype='int32', padding='post',
                                                             truncating='post', value=0.0)
print(padded_input)

embedding_layer = keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)

masked_output = embedding_layer(padded_input)
print('Shape of masked input passed through embedding: {}'.format(padded_input.shape))
print('Shape of Embeding output: {}'.format(masked_output.shape))
print('Shape of mask attached to this input: {}'.format(masked_output._keras_mask.shape))
print('Original input: \n{}'.format(padded_input))
print('attached Mask: \n{}'.format(masked_output._keras_mask))

# Using Masking layer:
masking_layer = keras.layers.Masking()
# simulate the embedding look up by expanding the 2D input to 3D
# with embedding dimension of 10
mask_value = 0.
x = np.random.rand(10, 6, 2)
x[0, 3:, :] = mask_value

print('\n\nUsing masking layer...')
print("Unmasked embedding: \n {}".format(x.shape))

masked_embedding = masking_layer(x)
print(masked_embedding._keras_mask.shape)
print(masked_embedding._keras_mask)

# Another example:
samples, timesteps, features = 32, 10, 8
x_train = np.random.random([samples, timesteps, features]).astype(np.float32)
y_train = np.random.random([samples, timesteps]).astype(np.float32)
x_train[:, 3, :] = mask_value
x_train[:, 5, :] = mask_value
y_train[:, 3] = mask_value
y_train[:, 5] = mask_value
y_masked = np.ones_like(y_train)
y_masked[y_train == 0.] = 0

input_tensor = keras.Input(shape=(timesteps, features))
masked_tensor = keras.Input(shape=(timesteps,))
# since the masked values are in the middle of sequence, need to make sure we correctly mask it before
# sending to LSTM:
# the outputs of masked steps will be skipped in LSTM calculations!!
masked_input = keras.layers.Masking(mask_value=mask_value)(input_tensor)
lstm_out = keras.layers.LSTM(10, return_sequences=True)(masked_input)
output = tf.squeeze(keras.layers.Dense(1)(lstm_out), axis=-1)
output *= masked_tensor
model = keras.Model(inputs=[input_tensor, masked_tensor], outputs=[output])
print(model.summary())


class MaskedMSE(keras.losses.Loss):
    """
    Assumption is that y_pred and y_true are already masked!
    If only y_true is masked, we need to multiply the mask to y_pred
    """
    def __init__(self, reduction=keras.losses.Reduction.AUTO,
                 name='Masked_MSE'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        mask = 1.0 - tf.cast(tf.equal(y_true, mask_value), tf.float32)
        y_pred *= mask
        se = tf.reduce_sum(tf.square(y_true - y_pred))
        mse = se / tf.reduce_sum(mask)
        return mse


loss = MaskedMSE(name='masked_mse')
model.compile(loss=loss, optimizer=keras.optimizers.Adam())

out = model([x_train, y_masked])
print(out.shape)
print(out[:, 3])

print(y_train[:, 3])
print('-'*10)
model.fit([x_train, y_masked], y_masked, epochs=2)
