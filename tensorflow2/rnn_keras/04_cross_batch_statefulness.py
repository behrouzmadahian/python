"""
When processing very long sequences (possibly infinite), you may want to use the pattern of cross-batch statefulness.

Normally, the internal state of a RNN layer is reset every time it sees a new batch 
(i.e. every sample seen by the layer is assume to be independent from the past). 
The layer will only maintain a state while processing a given sample.

If you have very long sequences though, it is useful to break them into shorter sequences, 
and to feed these shorter sequences sequentially into a RNN layer without resetting the layer's state.
That way, the layer can retain information about the entirety of the sequence,
even though it's only seeing one sub-sequence at a time.

You can do this by setting stateful=True in the constructor.

If you have a sequence s = [t0, t1, ... t1546, t1547], you would split it into e.g.
"""
import numpy as np
from tenrosflow import keras
import tensorflow as tf

paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

input = keras.Input(shape=(10, 50), name='input')
lstm_layer = keras.layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)
model = keras.Model(inputs=input, outputs=output)
print(model.summary())

# reset_states() will reset the cached state to the original initial_state.
# If no initial_state was provided, zero-states will be used by default.
lstm_layer.reset_states()
