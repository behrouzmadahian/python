"""
Recurrent neural networks (RNN) are a class of neural networks that is powerful for 
modeling sequence data such as time series or natural language.
Schematically, a RNN layer uses a for loop to iterate over the timesteps of a sequence, 
while maintaining an internal state that encodes information about the timesteps it has seen so far.

The Keras RNN API is designed with a focus on:
Ease of use: the built-in tf.keras.layers.RNN, tf.keras.layers.LSTM, tf.keras.layers.GRU layers 
enable you to quickly build recurrent models without having to make difficult configuration choices.

Ease of customization: You can also define your own RNN cell layer (the inner part of the for loop) 
with custom behavior, and use it with the generic tf.keras.layers.RNN layer (the for loop itself). 
This allows you to quickly prototype different research ideas in a flexible way with minimal code.

unit_forget_bias: Boolean (default True). If True, add 1 to the bias of the forget gate at initialization.
Setting it to true will also force bias_initializer="zeros". This is recommended in Jozefowicz et al..

Call arguments to RNN cells:
- inputs: A 3D tensor with shape [batch, timesteps, feature].
- mask: Binary tensor of shape [batch, timesteps] indicating whether a given timestep should be masked
        (optional, defaults to None).
- training: Python boolean indicating whether the layer should behave in training mode or in inference mode.
           This argument is passed to the cell when calling it. This is only relevant if dropout or recurrent_dropout is used
           (optional, defaults to None).
- initial_state: List of initial state tensors to be passed to the first call of the cell 
                (optional, defaults to None which causes creation of zero-filled initial state tensors).

In addition, a RNN layer can return its final internal state(s). The returned states can be used to 
resume the RNN execution later, or to initialize another RNN. This setting is commonly used in the 
encoder-decoder sequence-to-sequence model, where the encoder final state is used as the initial state of the decoder.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras

encoder_vocab = 1000
decoder_vocab = 2000
encoder_input = keras.Input(shape=(None,))
encoder_embedded = keras.layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)
# return states in addition to output:
output, state_h, state_c = keras.layers.LSTM(64, return_state=True, name='encoder')(encoder_embedded)
encoder_state = [state_h, state_c]
print('SHape of state_h and state_c : {}, {}'.format(state_h.shape, state_c.shape))

decoder_input = keras.Input(shape=(None,))
decoder_embedded = keras.layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)
# Pass the 2 states to a new LSTM layer as initial state:
decoder_output = keras.layers.LSTM(64, name='decoder')(decoder_embedded, initial_state=encoder_state)
output = keras.layers.Dense(10)(decoder_output)
model = keras.Model([encoder_input, decoder_input], outputs=output)
print(model.summary())
