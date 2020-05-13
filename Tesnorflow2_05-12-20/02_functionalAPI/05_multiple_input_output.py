"""
The functional API makes it easy to manipulate multiple inputs and outputs.
This cannot be handled with the Sequential API.
Let's say you're building a system for ranking custom issue tickets by priority
 and routing them to the right department.

You model will have 3 inputs:

1. Title of the ticket (text input)
2. Text body of the ticket (text input)
3. Any tags added by the user (categorical input)

It will have two outputs:

1. Priority score between 0 and 1 (scalar sigmoid output)
2. The department that should handle the ticket (softmax output over the set of departments)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
tf.keras.backend.clear_session()

num_tags = 12         # number of unique issue tags
num_words = 10000     # size of vocabulary
num_departments = 4   # unique number of departments

# Input layers:
title_input = keras.Input(shape=(None,), name='title')      # variable length sequence of int
body_input = keras.Input(shape=(None,), name='body')        # variable length sequence of ints
tags_input = keras.Input(shape=(num_tags,), name='tags')    # binary vector of size num_tags

# Embed each word in the title into a 64-dim vector
title_features = keras.layers.Embedding(num_words, 64)(title_input)
# Embed each word in the body of text into a 64 dim embedding:
body_features = keras.layers.Embedding(num_words, 64)(body_input)

# reduce sequence of embedded words in the title into a single 128-dim vector:
"""
unit_forget_bias=True: 
If True, add 1 to the bias of the forget gate at initialization.
Setting it to true will also force bias_initializer="zeros"
dropout: fraction of unit to drop
recurrent_dropout: fraction of units to drop for the linear transformation of the recurrent state, Default 0
implementation: 
implementation mode, either 1 or 2. 
Mode 1: will structure its operations as a larger number of smaller dot products and additions, 
Mode 2 will batch them into fewer, larger operations. 
These modes will have different performance profiles on different hardware and for different applications. 
Default: 2.
return_sequences: Whether to return the last output. in the output sequence, or the full sequence. Default: False.
return_state: Boolean. Whether to return the last state in addition to the output. Default: False.
"""
title_features = keras.layers.LSTM(128, unit_forget_bias=True, dropout=0.1,
                                   recurrent_initializer='orthogonal',
                                   return_sequences=False, time_major=False)(title_features, training=True)
# reduce sequence of embedded words in the text body into a single 32-dim vector:
body_features = keras.layers.LSTM(32, unit_forget_bias=True, dropout=0.1,
                                  recurrent_initializer='orthogonal',
                                  return_sequences=False, time_major=False)(body_features, training=True)

# merge all available features into a single layer vector via concatenation:
x = keras.layers.concatenate([title_features, body_features, tags_input])

# stick a logistic regression for priority prediction on top of the features:
priority_pred = keras.layers.Dense(1, activation=None, name='priority')(x)
# Stick a department classifier on top of the features
department_pred = keras.layers.Dense(num_departments, activation=None, name='department')(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(inputs=[title_input, body_input, tags_input], outputs=[priority_pred, department_pred])
print(model.summary())
checkpoint_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/chkpoint_files'

keras.utils.plot_model(model, checkpoint_folder+'/multiple_in_out_model_graph.png', show_shapes=True)

# when compiling this model, we can assign different losses to each output. and assign different weights to each loss
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss=[keras.losses.BinaryCrossentropy(from_logits=True),
                    keras.losses.CategoricalCrossentropy(from_logits=True)],
              loss_weights=[1.0, 0.2])
"""
Since we gave names to our output layers, we could also specify the loss like this:
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss={'priority': keras.losses.BinaryCrossentropy(from_logits=True),
                    'department': keras.losses.CategoricalCrossentropy(from_logits=True)},
              loss_weights=[1., 0.2])

"""
# create dummy data to train:
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))
print(priority_targets[:5])
print(dept_targets[:2])

model.fit({'title': title_data, 'body': body_data, 'tags': tags_data},
          {'priority': priority_targets, 'department': dept_targets},
          epochs=2,
          batch_size=32
          )

