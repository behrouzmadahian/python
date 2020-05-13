"""
When processing sequence data, it is very common for individual samples to have diff lengths
Since input data to model must be a single tensor of (batch, timesteps, f),
samples that are shorter than the longest item need to be padded with some placeholder
value,
# Padding the sequence with zeros- default behavior!:
raw_inputs: list of list! each sample is a list within this list!
padded_seq = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, maxlen=None,
                                                         dtype=int32, padding='post',
                                                         truncating='post', value=0)

maxlen: max sequence len
padding: pad the sequences at the beginning or the end
truncate: truncate sequences to maxlen, pre or post
value: padding value
Returns: numpy array with shape:
(batch, maxlen)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras

raw_inputs = [[83, 91, 1, 645, 1253, 927],
              [73, 8, 3215, 55, 927],
              [711, 632, 71]]
padded_seq = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, maxlen=None,
                                                           dtype='int32', padding='post',
                                                           truncating='post', value=0.0)
print(padded_seq)

"""
Let's understand the functionality of tf.pad in tensorflow 2.1

tf.pad(tensor, paddings, mode='CONSTANT', constant_values=0, name=None)

This operation pads a tensor according to the paddings you specify.
Paddings is an integer tensor with shape [n, 2], where n is the rank of tensor
For each dimension D of the input, 
paddings[D, 0] indicates how many values to add BEFORE the contents of tensor in that dimension, 
paddings[D, 1] indicates how many values to add AFTER the contents of tensor in that dimension. 
if mode is 'REJECT' then both paddings[D, 0] and paddings[D, 1] must be no greater than tensor.dim_size(D)-1
if mode is "SYMMETRIC" then both paddings[D, 0] and paddings[D, 1] mus be no greater than tensor.dim_size(D)

I do not quite understand how tf.pad works!!!!

"""
print('-' * 50)
t = tf.constant([[1, 2, 3], [4, 5, 6]])
print(t)
paddings = tf.constant([[0, 1], [0, 1]])
print(paddings)
padded_t = tf.pad(t, paddings, "CONSTANT", constant_values=0)
print(padded_t)
