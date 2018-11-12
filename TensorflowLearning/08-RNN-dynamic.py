import tensorflow as tf
import random
import numpy as np
'''
A dynamic Recurrent neural network using Tensorflow.
Used data: a toy data set to classify linear sequences. 
The generated sequences have variable length.
'''


# toy data generation
class ToySequenceData(object):
    def __init__(self, n_samples=1000, max_seq_length=20, min_seq_length=3, max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.batch_id = 0
        for i in range(n_samples):
            # random sequence length
            len = random.randint(min_seq_length, max_seq_length)
            # keep actual sequence length for dynamic calculations
            self.seqlen.append(len)
            # add a  linear (if P < 0.5) int sequence or random sequence (if P > 0.5)
            if random.random() < 0.5:
                # generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in range(rand_start, rand_start + len)]
                # pad sequence with 0's for dimension consistency in Tensorflow
                s += [[0.0] for i in range(max_seq_length - len)]
                self.data.append(s)
                self.labels.append([1.0, 0.0])
            else:
                # generate a random sequence
                s = [[float(random.randint(0, max_value)) / max_value] for i in range(len)]
                # pad sequence with 0's for dimension consistency in Tensorflow
                s += [[0.0] for i in range (max_seq_length - len)]
                self.data.append(s)
                self.labels.append([0.0, 1.0])

    def next(self, batch_size):
        ''' return  next batch of data. When dataset end is reached, start over. '''
        if self.batch_id == len(self.data):
            self.batch_id = 0
        data_batch = self.data[self.batch_id: min(self.batch_id + batch_size, len(self.data))]
        labels_batch = self.labels[self.batch_id: min(self.batch_id + batch_size, len(self.data))]
        seqlen_batch = self.seqlen[self.batch_id : min(self.batch_id + batch_size, len(self.data))]
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return data_batch, labels_batch, seqlen_batch


# model parameters:
learning_rate = 0.001
trainint_iters = 1000
batch_size = 128
display_step = 10
dropout = 0.5

# network parameters
seq_max_len = 20  # maximum length of sequences
n_hidden = 64     # hidden layer size
n_classes = 2

# generate data using the above toy sequence generator
train_set = ToySequenceData(n_samples=1000, max_seq_length=seq_max_len)
test_set = ToySequenceData(n_samples=500, max_seq_length=seq_max_len)
print(np.array(test_set.next(2)[0]).shape)

# tensorflow graph input
x = tf.placeholder(tf.float32, [None, seq_max_len, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)

# weights and biases dicts
weights = {
    'w': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'b': tf.Variable(tf.random_normal([n_classes]))
}


# wrapper for dynamic RNN
def dynamicRNN(x, seqlen, weights, biases, keep_prob):
    # prepare data for RNN. Current data shape is (batch_size, seq_max_len, n_input)
    # required data shape is seq_max_len tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, axis=1)
    # define LSTM cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # get LSTM cell output, providing 'sequence_length' performs dynamic calculation
    # it zeros out the output after the seq length point!!!
    lstm_out, states = tf.contrib.rnn.static_rnn(lstm_cell,
                                                 x,
                                                 initial_state=None,
                                                 dtype=tf.float32,
                                                 sequence_length=seqlen)

    # when performing dynamic calculation, we should get the last dynamically computed output,
    # i.e., if a sequence length is 10, we should get the 10th output.
    # 'outputs' is a list of output at every time step: [(batch_size, Output_1), ..].
    # We pack them in a Tensor and change
    # stack it to:  (seq_max_len, batch_size, n_hidden)
    # transpose  dimension to (batch_size, seq_max_len, n_output)

    # note: We did the padding at the end of each sequence!
    lstm_out = tf.stack(lstm_out)
    lstm_out = tf.transpose(lstm_out, [1, 0, 2])
    print('Shape of transposed and stacked LSTM output:', lstm_out.get_shape())
    # Hack to build indexing and get the right output:
    batch_size = tf.shape(lstm_out)[0]

    # end indices for each sample. if we flatten all samples into 20*batch_size
    # we can use this 1D indices to get the right output
    index = tf.range(batch_size) * seq_max_len + (seqlen - 1)
    # indexing now, outputs contains only the last unit of each sequence)
    print(tf.reshape(lstm_out, [-1, n_hidden]).get_shape())

    lstm_out = tf.gather(tf.reshape(lstm_out, [-1, n_hidden]), index, axis=0)
    print(lstm_out.get_shape())

    # drop out
    lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_prob)

    logits = tf.add(tf.matmul(lstm_out, weights['w']), biases['b'])

    return logits


logits = dynamicRNN(x, seqlen, weights, biases, keep_prob)

# loss and optimizer
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# evaluate model
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# initialize variables
init = tf.global_variables_initializer()

# launch default graph:
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < trainint_iters:
        x_batch, y_batch, seqlen_batch = train_set.next(batch_size)
        sess.run(optimizer, feed_dict={x: x_batch,
                                       y: y_batch,
                                       keep_prob: dropout,
                                       seqlen: seqlen_batch})
        # log every display step steps:
        if step % display_step == 0:
            batch_loss, batch_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={x: x_batch,
                                                             y: y_batch,
                                                             keep_prob: 1.0,
                                                             seqlen: seqlen_batch})
            print("Iteration:", (step * batch_size), "Batch cost=", batch_loss, "Batch accuracy=", batch_accuracy)

        step += 1

    print('Optimization Finished!')
    print ('Test model on test data')
    x_test, y_test, seqlen_test = test_set.next(500)
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test,
                                                  y: y_test,
                                                  keep_prob: 1.0,
                                                  seqlen: seqlen_test})
    print(test_accuracy)



