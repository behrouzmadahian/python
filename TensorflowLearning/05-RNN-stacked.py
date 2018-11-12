import tensorflow as tf

'''
Two  layer Recurrent neural network
note:
the state has tuple of tuples: ( (h1, c1), (h2, c2) )

data Mnist data of hand written digits. 28*28 images of 1 channel
to apply RNN to this data we assume, col i of each image is set of features at time step i

Note:
in tf.contrib.rnn.static_rnn:
requires n-step tensors of shape (batch_size, n_features)
i.e: we need to convert our data into list of these tensors from timestep 1 to 28

output:
returns a list of outputs. element i is:
output of shape (batch_size, n_hidden) representing the output at timestep i of the FINAL LAYER!

Note:
if keeping the states between batches is desired, need to define the state out of wrapper function
as placeholder and feed it as initial state.
'''
from tensorflow.examples.tutorials.mnist import input_data
train, validation, test = input_data.read_data_sets('./mnist', one_hot=True)


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


# use this for initializing biases if using relu activation
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


# parameters:
learning_rate = 0.001
training_iters = 10000
batch_size = 64
dropout = 0.5  # keep probability!
display_step = 10
l2Reg = 0.0001

# network parameters:
n_feat = 28
time_steps = 28
n_hidden = 128
n_classes = 10
n_layers = 2

# tensorflow graph inputs
x = tf.placeholder(tf.float32, [None, time_steps, n_feat])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# dictionaries for weights and biases of output layer
weights = {
    'w': weight_variable([n_hidden, n_classes], name='W_out')
}
biases = {
    'b': bias_variable([n_classes], name='B_out')
}


# wrapper for RNN
def stacked_rnn (x, weights, biases, keep_prob, l2Reg=0.):

    # since I use tf.contrib.rnn.static_rnn:
    # need to convert input of shape (batch_size, time_steps,features)
    # to list of (batch_size, features) of length time_steps

    x = tf.unstack(x, time_steps, axis=1)

    def lstm_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.tanh)
        # if applying dropout after each layer is NOT desired, remove this and
        # maybe apply drop out to the output of the stack!
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        return cell

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
    initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
    lstm_out, states = tf.contrib.rnn.static_rnn(stacked_lstm, x, initial_state=initial_state)

    print('Shape of output tensor:', len(lstm_out), lstm_out[0].get_shape())
    print('shape of state tensor: ( (h1, c1), (h2, c2),.. ) '
          '  the final state (c_i) of shape (batch_size, n_hidden)=',
          len(states), states[1][0].get_shape())

    # if no dropout applied at each layer and applying at final layer output is desired
    # lstm_out= tf.nn.dropout(lstm_out[-1], keep_prob=dropout)
    logits = tf.add(tf.matmul(lstm_out[-1], weights['w']), biases['b'])
    # l2Reg: chose to not regularize output layer
    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_out' in curr_var.name))
    l2Loss = tf.scalar_mul(l2Reg, l2Loss)
    return logits, l2Loss


logits, l2Loss = stacked_rnn(x, weights, biases, keep_prob, l2Reg=0.)
data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
total_loss = tf.add(data_loss, l2Loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initializing variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < training_iters:
        x_batch, y_batch = train.next_batch(batch_size)
        x_batch = x_batch.reshape([-1, time_steps, n_feat])
        sess.run(optimizer, feed_dict={x: x_batch,
                                       y: y_batch,
                                       keep_prob: dropout})

        if step % display_step == 0:
            batch_data_loss, batch_l2_Loss, batch_total_loss, batch_accuracy = \
                sess.run([data_loss, l2Loss, total_loss, accuracy],
                         feed_dict={x: x_batch,
                                    y: y_batch,
                                    keep_prob: 1.0})

            print('Iteration', step, 'Batch data loss=', batch_data_loss,
                  'batch Regularization loss=', batch_l2_Loss,
                  'batch total loss=', batch_total_loss, 'Batch accuracy=', batch_accuracy)
        step += 1
    print('Optimization finished')
    xtest, y_test = test
    xtest = xtest.reshape([-1, time_steps, n_feat])
    test_loss, test_accuracy = sess.run([data_loss, accuracy], feed_dict={x: xtest,
                                                                          y: y_test,
                                                                          keep_prob: 1.0})
    print('Test loss=', test_loss, 'Test Accuracy=', test_accuracy)