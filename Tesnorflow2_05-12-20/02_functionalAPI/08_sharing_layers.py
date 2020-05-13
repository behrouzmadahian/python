"""
Shared layers are layer instances that get reused multiple times in a same model: they learn features
that correspond to multiple paths in the graph-of-layers.
Shared layers are often used to encode inputs that come from similar spaces
(say, two different pieces of text that feature similar vocabulary),
since they enable sharing of information across these different inputs, and they make it possible to train such
a model on less data. If a given word is seen in one of the inputs, that will benefit the processing of all inputs
that go through the shared layer.

To share a layer in the Functional API, just call the same layer instance multiple times.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

# Embedding for 1000 unique words mapped to 128 dim vectors
shared_embedding = keras.layers.Embedding(1000, 128)

# Variable length sequence of integers:
text_input_a = keras.Input(shape=(None,), dtype='int32')
# Variable length sequence of integers:
text_input_b = keras.Input(shape=(None,), dtype='int32')
# reuse the same layer to encode the integers:
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)
