"""
In addition to the built-in RNN layers, the RNN API also provides cell-level APIs. 
Unlike RNN layers, which processes whole batches of input sequences, the RNN cell only processes a single timestep.

The cell is the inside of the for loop of a RNN layer. Wrapping a cell inside a tf.keras.layers.RNN layer
gives you a layer capable of processing batches of sequences, e.g. RNN(LSTMCell(10)).

Mathematically, RNN(LSTMCell(10)) produces the same result as LSTM(10).
In fact, the implementation of this layer in TF v1.x was just creating the corresponding RNN cell and wrapping it 
in a RNN layer. However, using the built-in GRU and LSTM layers enables the use of CuDNN and you may see better performance.

There are three built-in RNN cells, each of them corresponding to the matching RNN layer.

tf.keras.layers.SimpleRNNCell corresponds to the SimpleRNN layer.

tf.keras.layers.GRUCell corresponds to the GRU layer.

tf.keras.layers.LSTMCell corresponds to the LSTM layer.

The cell abstraction, together with the generic tf.
"""
