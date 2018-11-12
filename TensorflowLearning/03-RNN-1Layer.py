import tensorflow as tf

'''
A recurrent Neural network (LSTM) using Tensorflow
used data: MNIST databsase of handwritten digits
We use BasicLSTMCell:
It does not allow cell clipping, a projection layer, and does not use peep-hole connections.
Note:
we set forget bias=1.0  in order to reduce the scale of forgetting in the beginning of the training.
Must set to 0.0 manually when restoring from CudnnLSTM-trained checkpoints.

rnn.static_rnn:
an initial state can be provided.
[batch_size, cell.state_size]

the states returned  are final states! and are tuple of length 2 of shape above
( hidden_states<h>, cell states<c>  ) 
output:
returns a list of outputs. element i is:
output of shape (batch_size, n_hidden) representing the output at timestep i

Note: 
the state is reset after each batch!
'''


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

# to use RNN, we consider every image row as a sequence of pixels.
# MNIST images are 28x28 pixels, so we have 28 sequences (columns) of 28 time steps (rows) for every sample.

# parameters:
learning_rate = 0.001
training_iterations = 1000000
batch_size = 128
display_step = 10  # display resutls every 10 steps
dropout = 0.5

# network parameters:
n_features = 28    # mnist data rows
n_steps = 28         # mnist data columns(timesteps)
n_hidden = 128
n_classes = 10
reg = 0.0002

# Placeholders for inputs to feed_dict
x = tf.placeholder(tf.float32, [None, n_steps, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# defining weight variables and biases: for the final output layer!
weights = {
    'w': weight_variable([n_hidden, n_classes], name='W_out')
}
biases = {
    'b': bias_variable([n_classes], name='B_out')
}

# wrapper for our RNN:


def RNN(x, weights, biases, keep_prob):

    # Prepare data for RNN.
    # original data is [batch_size, n_steps*n_features].
    # we reshaped it to: (batch_size, n_steps, n_features) before feeding to model.
    # required data shape for RNN is n_steps tensors of shape(batch_size, n_features)
    # i.e. we need to prepare the data as list of features for each time steps,
    # if say we have 10 time steps, then we want to have list of length 10 with each element being
    # a matrix of [Batch_size, n_features] for that time step.

    x = tf.unstack(x, n_steps, axis=1)
    # define the LSTM cell:
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.tanh)

    # get LSTM output:
    lstm_out, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, initial_state=None)
    print('shape of last output of the LSTM layer', lstm_out[-1].get_shape())
    print('shape of last hidden state h', states[0].get_shape())
    print('shape of last cell state c', states[1].get_shape())

    # dropout:
    lstm_out = tf.nn.dropout(lstm_out[-1], keep_prob=keep_prob)

    # output layer:
    # note: we get the last hidden state of LSTM from each node!
    logits = tf.add(tf.matmul(lstm_out, weights['w']), biases['b'])

    # Add regularization-l2 loss:
    # We dont regularize last layer's weights and biases here!
    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_out' in curr_var.name))

    return logits, l2Loss


# prediction model
logits, l2Loss = RNN(x, weights, biases, keep_prob)
l2Loss *= reg

# loss function and optimization definitions
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
loss_plus_reg = tf.add(loss, l2Loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_plus_reg)

# evaluate model
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialization - to be run in session
init = tf.global_variables_initializer()

# launch default graph:
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iterations:
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch = x_batch.reshape([-1, n_steps, n_features])
        sess.run(optimizer, feed_dict={x: x_batch, y: y_batch, keep_prob: dropout})
        if step % display_step == 0:
            batch_cost, batch_accuracy, RegLoss = sess.run([loss, accuracy, l2Loss],
                                                           feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})

            print('Iteration', step, 'Batch data loss=', batch_cost,
                  'batch Regularization loss=', RegLoss, 'Batch accuracy=', batch_accuracy)

        step += 1
    print('Optimization finished!')

    # testing the model:
    x_test = mnist.test.images.reshape((-1, n_steps, n_features))
    y_test = mnist.test.labels
    print('Test accuracy: ')
    print(sess.run(accuracy,
                   feed_dict={x: x_test, y: y_test, keep_prob: 1.0}))



