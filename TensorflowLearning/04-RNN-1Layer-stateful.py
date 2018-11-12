import tensorflow as tf
import numpy as np

'''
Single layer stateful Recurrent neural network
--state is NOT reset between batches--

last state from batch i sample j will be used as initial state for
bacth i+1 sample j

data Mnist data of hand written digits. 28*28 images of 1 channel
to apply RNN to this data we assume, col i of each image is set of features at time step i

The Mnist data is not the best data to work with as state needs to be reset between batches

Note:
tf.contrib.rnn.static_rnn: requires n-step tensors of shape (batch_size, n_features)
i.e: we need to convert our data into list of these tensors from time step 1 to 28
SINCE the batchsize has to be defined in defining the shape of the placeholder for states,
at the time of testing and training, batches of size batch_size must be fed in!.
The work around is defining placeholder for state, use the first dimension None! 
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


# a function used to generate initial state for feed_dict
def init_state(batch_size, n_hidden):
    c_state = np.zeros([batch_size, n_hidden])
    h_state = np.zeros([batch_size, n_hidden])
    state = (h_state, c_state)
    return state


# parameters:
learning_rate = 0.001
training_iters = 10
batch_size = 64
dropout = 0.5
display_step = 10
l2Reg = 0.0

# network parameters:
n_feat = 28
time_steps = 28
n_hidden = 128
n_classes = 10

# tensorflow graph inputs:
x = tf.placeholder(tf.float32, [None, time_steps, n_feat])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# define state of the network so we can feed it in feed_dict
hidden_state = tf.placeholder(tf.float32, [None, n_hidden])
current_state = tf.placeholder(tf.float32, [None, n_hidden])
states = hidden_state, current_state

# define weights and biases of output layer
weights = {
    'w': weight_variable([n_hidden, n_classes], name='W_out')
}
biases = {
    'b': bias_variable([n_classes], name='B_out')
}


def rnn(x, weights, biases, states, keep_prob, l2Reg=0.0):
    # when using tf.contrib.rnn.static_rnn:
    # need to convert input of shape (batch_size, time_steps, features)
    # to list of (batch_size, features) of length time_steps
    x = tf.unstack(x, time_steps, axis=1)  # unstack axis 1
    # define LSTM cell:
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.tanh)

    # get lstm state and output
    # outputs has the shape : (time_steps, batch_size, n_hidden)
    # we choose to use last output

    lstm_out, states1 = tf.contrib.rnn.static_rnn(lstm_cell, x, initial_state=states, dtype=tf.float32)

    print('shape of LSTM output tensor at final time step=', lstm_out[-1].get_shape())

    print('shape of state tensor: a tupple of length %d each having:'
          '  the final state of shape (batch_size, n_hidden)=' % len(states1), states1[1].get_shape())
    # dropout
    lstm_out = tf.nn.dropout(lstm_out[-1], keep_prob=keep_prob)
    # linear activation
    logits = tf.add(tf.matmul(lstm_out, weights['w']), biases['b'])

    # l2 regularization- we decide not to regularize output layer
    l2loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_out' in curr_var.name)
                 )
    l2loss = tf.scalar_mul(l2Reg, l2loss)
    return logits, l2loss, states1


logits, l2loss, batch_end_state = rnn(x, weights, biases, states, keep_prob, l2Reg)

# loss and optimizer
dataLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
total_loss = dataLoss + l2loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

# evaluation
correct_preds = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# initialization
init = tf.global_variables_initializer()
# launching default graph:
with tf.Session() as sess:
    # getting the value of initial state:
    numpy_batch_end_state = init_state(batch_size, n_hidden)
    sess.run(init)
    step = 1
    while step < training_iters:
        x_batch, y_batch = train.next_batch(batch_size)
        x_batch = x_batch.reshape((-1, time_steps, n_feat))
        _, numpy_batch_end_state = sess.run([optimizer, batch_end_state],
                                            feed_dict={x: x_batch,
                                                       y: y_batch,
                                                       keep_prob: dropout,
                                                       states: numpy_batch_end_state
                                                       })
        if step % display_step == 0:
            batch_data_loss, batch_l2_loss, batch_total_loss, batch_accuracy = \
                sess.run([dataLoss, l2loss, total_loss, accuracy],
                         feed_dict={x: x_batch,
                                    y: y_batch,
                                    keep_prob: 1.0,
                                    states: numpy_batch_end_state})

            print('Batch %d: data loss=' % step, batch_data_loss, 'l2 loss=', batch_l2_loss,
                  'total loss=', batch_total_loss, 'batch accuracy=', batch_accuracy)

        step += 1
    print('finished! Optimization was run for %d batches' % step)

    print('Test loss and accuracy: ')
    # x_test, y_test = test.next_batch(batch_size)
    x_test, y_test = test.images, test.labels
    x_test = x_test.reshape((-1, time_steps, n_feat))
    numpy_init_state = init_state(x_test.shape[0], n_hidden)

    print(sess.run([dataLoss, accuracy], feed_dict={x: x_test,
                                                    y: y_test,
                                                    keep_prob: 1.0,
                                                    states: numpy_init_state}))