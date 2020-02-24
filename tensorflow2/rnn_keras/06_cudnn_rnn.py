"""
In TensorFlow 2.0, the built-in LSTM and GRU layers have been updated to leverage CuDNN kernels by default 
when a GPU is available. With this change, the prior keras.layers.CuDNNLSTM/CuDNNGRU layers have been deprecated, 
and you can build your model without worrying about the hardware it will run on.

Since the CuDNN kernel is built with certain assumptions, this means the layer will not be able to use the CuDNN kernel 
if you change the defaults of the built-in LSTM or GRU layers. E.g.:

- Changing the activation function from tanh to something else.
- Changing the recurrent_activation function from sigmoid to something else.
- Using recurrent_dropout > 0.
- Setting unroll to True, which forces LSTM/GRU to decompose the inner tf.while_loop into an unrolled for loop.
- Setting use_bias to False.
- Using masking when the input data is not strictly right padded 
  (if the mask corresponds to strictly right padded data, CuDNN can still be used. This is the most common case).

For the detailed list of constraints, please see the documentation for the LSTM and GRU layers.

Using CuDNN kernels when available
Let's build a simple LSTM model to demonstrate the performance difference.

We'll use as input sequences the sequence of rows of MNIST digits (treating each row of pixels as a timestep),
and we'll predict the digit's label.
"""
