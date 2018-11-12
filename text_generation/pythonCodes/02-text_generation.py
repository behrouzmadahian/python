import tensorflow as tf
import numpy as np
import random
import os
'''
We will use GloVe pre-trained word embeddings which have length =50
The word2Vec has 3 million words of each 300 length! a little memory intensive!
Glove contains 400000 words with word vector representation of length 50!
The training set we're going to use is the Imdb movie review dataset. 
This set has 25,000 movie reviews, with 12,500 positive reviews and 12,500 negative reviews.
Each of the reviews is stored in a txt file that we need to parse through. 
we make sure the first half of test data is  positive and the second half is negative samples

# now that we have our vectors, first step is taking an input sentence and then constructing its vector representation.
#assume we have the sentence:
#i thought the movie was incredible and inspiring.
# in order to get the word vectors, we can use tensorflow's embedding lookup function.
#this function take two arguments, one for embedding matrix, the other, the id;s of each of the words.
#the ids vector can be thought of intigerized  representation of the training set.
#this is basically just the row index of each of the words.
max- seq length must be determined in the preprocessing file!
'''

vec_length = 50
batch_size = 256
seq_len = 30
lstmUnits = 64
# dimension of word vectors
dropout = 0.2
init_lr = 0.005
data_dir = 'C:/behrouz/PythonCodes/ML/text_generation/'
datax = np.load(data_dir + 'datax.npy')
datay = np.load(data_dir + 'datay.npy')
print(datax.shape, datay.shape)
vocab_list = np.loadtxt(data_dir + 'vocabulary_list.txt', delimiter='\t', dtype=str)
n_classes = len(vocab_list)
vocab_size = len(vocab_list)
print('Number of Unique words= ', len(vocab_list))
'''
Helper Functions:
Below you can find a couple of helper functions that will be useful when training the network in a later step
'''


def shuffle(data):
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)  # in-place shuffling
    data = data[inds]
    return data


# add the labels data for train and get the labels for BATCH!!!!
def getTrainBatch(datax, datay, batch_size):
    start = batch_id * batch_size
    end = min((batch_id + 1) * batch_size, datax.shape[0])
    batchx = datax[start: end]
    batchy = datay[start: end]
    return batchx, batchy


def oh_labels(y, num_classes):
    y = y.astype(int)
    ohm = np.zeros((len(y), num_classes), dtype=np.int32)
    ohm[np.arange(y.shape[0]), y] = 1
    return ohm


#######################################################################################################################
x = tf.placeholder(tf.int32, [None, seq_len])
y = tf.placeholder(tf.float32, [None, n_classes])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

weights = {
    'wordVecs': tf.get_variable(name='wordVecs', shape=[vocab_size, vec_length],
                                initializer=tf.contrib.layers.xavier_initializer()),
    'w': tf.get_variable(name='w', shape=[lstmUnits, n_classes],
                         initializer=tf.contrib.layers.xavier_initializer())
         }
biases = {
    'b': tf.Variable(tf.constant(0., shape=[n_classes]))
        }
'''
Once we have our input data placeholder, we’re going to call the
tf.nn.embedding_lookup() function in order to get our word vectors. 
The call to that function will return a 3-D Tensor of dimensionality batch size
 by max sequence length by word vector dimensions.
'''


# wrapper for dynamic RNN
def rnn(x, weights, biases, keep_prob):
    # x is [batch_size, seq_max_len], fist we need to get the embeddings
    print('Shape of input before embedding lookup= ', x.get_shape())
    x = tf.nn.embedding_lookup(weights['wordVecs'], x)
    print(x.get_shape())
    x = tf.unstack(x, seq_len, axis=1)

    print('Shape of embedding= ', len(x), x[0].get_shape())
    # prepare data for RNN. Current data shape is (batch_size, seq_max_len, n_input)
    # required data shape is seq_max_len tensors of shape (batch_size, n_input)
    # define LSTM cell

    def lstm_cell():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, forget_bias=1.0)
        # lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)
        return lstm_cell

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])
    # get LSTM cell output, providing 'sequence_length' performs dynamic calculation
    lstm_out, states = tf.contrib.rnn.static_rnn(stacked_lstm, x, initial_state=None,
                                                 dtype=tf.float32)
    print('Shapeof output of LSTM= ', len(lstm_out), lstm_out[-1].get_shape())
    lstm_out = lstm_out[-1]
    logits = tf.matmul(lstm_out, weights['w']) + biases['b']
    return logits


logits = rnn(x, weights, biases, keep_prob)
# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#  evaluate model
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
tf.summary.scalar('Learning-rate', learning_rate)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # train_summary_writer = tf.summary.FileWriter('tensorboard-vis/train', sess.graph)
    train_summary_writer = tf.summary.FileWriter('C:/behrouz/PythonCodes/ML/sentiment-analysis/results/tensorboard-vis/train')
    inds = np.arange(datax.shape[0])
    for epoch in range(10000):
        print('-'*100)
        print('EPOCH= ', epoch+1)
        np.random.shuffle(inds)
        datax = datax[inds]
        datay = datay[inds]
        batch_id = 0
        while batch_id * batch_size < datax.shape[0]:
            # Next Batch of reviews
            nextBatch, nextBatchLabels = getTrainBatch(datax, datay, batch_size)
            batch_id += 1
            nextBatchLabels_oh = oh_labels(nextBatchLabels, n_classes)

            sess.run(optimizer, feed_dict={x: nextBatch,
                                           y: nextBatchLabels_oh,
                                           learning_rate: init_lr,
                                           keep_prob: dropout})
            # Write summary to Tensorboard
            if (batch_id+1) % 100 == 0:
                train_summary, train_loss, train_accu = sess.run([merged, loss, accuracy],
                                                                 feed_dict={x: datax,
                                                                            y: oh_labels(datay, n_classes),
                                                                            learning_rate: init_lr,
                                                                            keep_prob: 1.})
                print('Train data Iteration=', batch_id, 'Loss=', train_loss, 'Accuracy=', train_accu)

    saver.save(sess, "C:/behrouz/PythonCodes/ML/text_generation/textgenModel.ckpt")
    train_summary_writer.close()


