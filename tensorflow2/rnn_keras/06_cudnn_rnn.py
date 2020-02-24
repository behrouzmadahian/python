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
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
batch_size = 64
# each MNISt digit image batch is a tensorf of shape:  (batch_size, 28, 28)
# each input sequence will be of size (28, 28); height is treated like time
input_dim = 28
units = 64 
output_size = 10 # labels from 0 to 9


def build_model(allow_cudnn_kernel=True):
  # CuDNN is only available at the layer level, and not at the cell level.
  # This means 'LSTM(units)' will use the CuDNN kernel,
  # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
  if allow_cudnn_kernel:
    # The LSTM layer with default options uses CuDNN
    lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
  else:
    # Wrapping a LSTMCell in a RNN layer will not use CuDNN
    lstm_layer =  keras.layers.RNN(keras.layers.LSTMCell(units), 
                                   input_shape=(None, input_dim))
  model = keras.models.Seauenctial([lstm_layer,
                                    keras.layers.BatchNormalization()
                                    keras.layers.Dense(output_size)])
  return model

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.
sample, sample_label = x_train[0], y_train[0]
print('Shape opf train x: {}'.format(trainx.shape))
model = build_model (allow_cuddn_kernel=True)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=5)

# build a new model without CuDNN kernel:
slow_model = build_model(allow_cudnn_kernel=False)
slow_model.set_weights(model.get_weights())
slow_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                   optimizer='sgd', 
                   metrics=['accuracy'])
slow_model.fit(x_train, y_train, 
               validation_data=(x_test, y_test), 
               batch_size=batch_size,
               epochs=1)  # We only train for one epoch because it's slower.

"""
The same CuDNN-enabled model can also be use to run inference in a CPU-only environment. 
The tf.device annotation below is just forcing the device placement.
The model will run on CPU by default if no GPU is available.
You simply don't have to worry about the hardware you're running on anymore. Isn't that pretty cool?
"""

with tf.device('CPU:0'):
  cpu_model = build_model(allow_cudnn_kernel=True)
  cpu_model.set_weights(model.get_weights())
  result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
  print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
  plt.imshow(sample, cmap=plt.get_cmap('gray'))
              
