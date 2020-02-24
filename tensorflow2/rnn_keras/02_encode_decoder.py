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
