import tensorflow as tf
'''
A bidirectional recurrent neural network (LSTM) using Tensorflow.
Used data: MNIST database of handwritten digits.

The initial state for both directions is zero by default (but can be set optionally) ->
 stateful across batches RNN
 
static_bidirectional_rnn()
returns:
 (outputs, output_state_fw, output_state_bw)
outputs: 
returns a list of outputs. element i is:
output of shape (batch_size, n_hidden*2) representing the output at time step i
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
train, validation, test = input_data.read_data_sets('./mnist', one_hot=True)

# to use RNN, we consider every image row as a sequence of pixels.
# MNIST images are 28x28 pixels, so we have 28 sequences (columns)
# of 28 time steps (rows) for every sample.

# parameters:
learning_rate = 0.001
training_iterations = 1000000
batch_size = 128
display_step = 1  # display resutls every 10 steps
dropout = 0.5
# network parameters:
n_features = 28    # mnist data rows
n_steps = 28       # mnist data columns(timesteps)
n_hidden = 128
n_classes = 10
reg = 0.0002
# Placeholders for inputs to feed_dict
x = tf.placeholder(tf.float32, [None, n_steps, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {
    # hidden layer weights: 2 * n_hidden (because of forward and backward cells)
    'w': weight_variable([2 * n_hidden, n_classes], name='W_out')
}
biases = {
    'b': bias_variable([n_classes], name='B_out')
}

# wrapper for bi-directional RNN
def birnn(x, weights, biases, keep_prob):
    '''
     # Prepare data for RNN.
     # original data is [batch_size, n_steps*n_features].
     #  we reshaped it to: (batch_size, n_steps, n_features) before feeding to model.
     # required data shape for tf.contrib.rnn.static_rnn() is n_steps tensors of shape(batch_size, n_features)
     # i.e. we need to prepare the data as list of features for each time steps,
     # if say we have 10 time steps, then we want to have list of length 10 with each element being
     # a matrix of [Batch_size, n_features] for that time step.
    '''
    x = tf.unstack(x, n_steps, axis=1)
    # define LSTM cells (forward and backward cells)
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,
                                                forget_bias=1.0,
                                                activation=tf.nn.tanh)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,
                                                forget_bias=1.0,
                                                activation=tf.nn.tanh)
    # get LSTM cell output:
    try:
        lstm_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                 lstm_bw_cell,
                                                                 x,
                                                                 dtype=tf.float32,
                                                                 initial_state_fw=None,
                                                                 initial_state_bw=None)

    except Exception: # old Tensorflow version only returns outputs not states
        lstm_out = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                           lstm_bw_cell,
                                                           x,
                                                           dtype=tf.float32,
                                                           initial_state_fw=None,
                                                           initial_state_bw=None)
    # dropout
    lstm_out = tf.nn.dropout(lstm_out[-1], keep_prob=keep_prob)
    print('output shape of bi directional LSTM=', lstm_out[-1].get_shape())
    logits = tf.add(tf.matmul(lstm_out, weights['w']), biases['b'])
    # l2Regularization loss:
    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'b_out' in curr_var.name))
    return logits, l2Loss


logits, l2Loss = birnn(x, weights, biases, keep_prob)

l2Loss *= reg
# loss and optimizer:
dataLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
dataLoss_plus_regLoss = tf.add(dataLoss, l2Loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(dataLoss_plus_regLoss)

# evaluate model
correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# initialization
init = tf.global_variables_initializer()
# launch default graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < training_iterations:
        x_batch, y_batch = train.next_batch(batch_size)
        x_batch = x_batch.reshape([-1, n_steps, n_features])
        sess.run(optimizer, feed_dict={x: x_batch,
                                       y: y_batch,
                                       keep_prob: dropout})

        if step % display_step == 0:
            batch_dataLoss, batchL2Loss, batchTotLoss, batch_accuray = \
                sess.run([dataLoss, l2Loss, dataLoss_plus_regLoss, accuracy],
                         feed_dict={x: x_batch,
                                    y: y_batch,
                                    keep_prob: 1.0})
            print('Iteration=', step, 'dataLoss=', batch_dataLoss, 'RegLoss=',
                  batchL2Loss, 'TotalLoss=', batchTotLoss, 'batch Accuracy=', batch_accuray)
        step += 1
    print("Optimization finished!")
    # test model on 128 test data
    x_test = test.images[:128].reshape((-1, n_steps, n_features))
    y_test = test.labels[:128]
    print("Test accuracy=", sess.run(accuracy, feed_dict={x: x_test,
                                                          y: y_test,
                                                          keep_prob: 1.0}))