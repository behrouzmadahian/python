"""
RNNs with list/dict inputs, or nested inputs
Nested structures allow implementers to include more information within a single timestep. 
For example, a video frame could have audio and video input at the same time. The data shape in this case could be:

[batch, timestep, {"video": [height, width, channel], "audio": [frequency]}]

In another example, handwriting data could have both coordinates x and y for the current position of the pen, 
as well as pressure information. So the data representation could be:

[batch, timestep, {"location": [x, y], "pressure": [force]}]

The following code provides an example of how to build a custom RNN cell that accepts such structured inputs.

Define a custom cell that support nested input/output
Any gating mechanism should be implemented in the call method and associated weihgts MUST be defined in __init__
"""
from tensorflow import keras
import tensorflow as tf
import collections
import numpy as np

nestedInput = collections.namedtuple('nestedInput', ['feature1', 'feature2'])
print(nestedInput)

nextedState = collections.namedtuple('nestedState', ['state1', 'state2'])


class NestedCell(keras.layers.Layer):
  def __init__(self, unit1, unit2, unit3, **kwargs):
    self.unit1 = unit1
    self.unit2 = unit2
    self.unit3 = unit3
    self.state_size = nextedState(state1=unit1, state2=tf.TensorShape([unit2, unit3]))
    self.output_size = (unit1, tf.TensorShape([unit2, unit3]))
    super(NestedCell, self).__init__(**kwargs)
  
  def build(self, input_shapes):
    # expect input_shape to contain 2 items, [(batch, i1), (batch, i2, i3)]
    input_1 = input_shapes.feature1[1]
    input_2, input_3 = input_shapes.feature2[1:]

    self.kernel_1 = self.add_weight(
        shape=(input_1, self.unit1), initializer='uniform', name='kernel_1')
    self.kernel_2_3 = self.add_weight(
        shape=(input_2, input_3, self.unit2, self.unit3),
        initializer='uniform',
        name='kernel_2_3')

  def call(self, inputs, states):
    # inputs should be in [(batch, input_1), (batch, input_2, input_3)]
    # state should be in shape [(batch, unit_1), (batch, unit_2, unit_3)]
    input_1, input_2 = tf.nest.flatten(inputs)
    s1, s2 = states

    output_1 = tf.matmul(input_1, self.kernel_1)
    output_2_3 = tf.einsum('bij,ijkl->bkl', input_2, self.kernel_2_3)
    state_1 = s1 + output_1
    state_2_3 = s2 + output_2_3

    output = [output_1, output_2_3]
    new_states = nextedState(state1=state_1, state2=state_2_3)

    return output, new_states
                                                
                                                  
unit_1 = 2
unit_2 = 2
unit_3 = 2

input_1 = 4
input_2 = 4
input_3 = 4
batch_size = 64
num_batch = 100
timestep = 50

cell = NestedCell(unit_1, unit_2, unit_3)
rnn = tf.keras.layers.RNN(cell)

inp_1 = tf.keras.Input((None, input_1))
inp_2 = tf.keras.Input((None, input_2, input_3))

outputs = rnn(nestedInput(feature1=inp_1, feature2=inp_2))

model = tf.keras.models.Model([inp_1, inp_2], outputs)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print(model.summary())
input_1_data = np.random.random((batch_size * num_batch, timestep, input_1))
input_2_data = np.random.random((batch_size * num_batch, timestep, input_2, input_3))
target_1_data = np.random.random((batch_size * num_batch, unit_1))
target_2_data = np.random.random((batch_size * num_batch, unit_2, unit_3))
input_data = [input_1_data, input_2_data]
target_data = [target_1_data, target_2_data]

model.fit(input_data, target_data, batch_size=batch_size)
