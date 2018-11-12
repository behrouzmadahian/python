import tensorflow as tf
import numpy as np
import random
'''
We will use GloVe pre-trained word embeddings which have length =50
The word2Vec has 3 million words of each 300 length! a little memory intensive!
Glove contains 400000 words with word vector representation of length 50!
The training set we're going to use is the Imdb movie review dataset. 
This set has 25,000 movie reviews, with 12,500 positive reviews and 12,500 negative reviews.
Each of the reviews is stored in a txt file that we need to parse through. 
we make sure the first half of test data is  positive and the second half is negative samples
'''
'''
# now that we have our vectors, first step is taking an input sentence and then constructing its vector representation.
#assume we have the sentence:
#i thought the movie was incredible and inspiring.
# in order to get the word vectors, we can use tensorflow's embedding lookup function.
#this function take two arguments, one for embedding matrix, the other, the id;s of each of the words.
#the ids vector can be thought of intigerized  representation of the training set.
#this is basically just the row index of each of the words.
'''
'''
max- seq length must be determined in the preprocessing file!
'''
maxSeqLength = 250
batch_size = 24
lstmUnits = 64
numClasses = 2
iterations = 5000
numDimensions = 50 # dimension of word vectors
dropout = 0.5
init_lr = 0.001
# IF VALIDATION ACCURACY DOES NOT IMPROVE FOR 200 BATCHES DECREASE LEARNING RATE
lr_decrease_patience = 10
lr_decrease_factor = 0.8
# if validation accuracy does not improve for 1000 batches, stop the training
early_stop_patience = 20

wordsList = np.load('training_data/wordsList.npy')
wordsList = wordsList.tolist()
# encode words as UTF-8
wordsList = [word.decode('UTF-8') for word in wordsList]
print('first 10 words in our wordslist=(UTF-8)', wordsList[:10])
# loading word vectors:
wordVectors = np.load('training_data/wordVectors.npy')
ids = np.load('idsMatrix.npy')
seq_lengths = np.load('file_lengths.npy').astype('int32')
# Pass into embedding function and see if it evaluates.
print('Length of word list=',len(wordsList))
print('Shape of word vector matrix=',wordVectors.shape)
print('Shape of id-s converted data=',ids.shape)
print('Shape of sequence length vector=',seq_lengths.shape)
print(seq_lengths[:20])

'''
Helper Functions:
Below you can find a couple of helper functions that will be useful when training the network in a later step
'''


def train_test_split(data, seq_lengths, p_end_ind, train_frac=0.9):
    '''
    I assume the first part of index data is positive up to index p_end_ind
    train_frac: fraction of positive and negative examples to take as part of train data!
    shuffles the data and returns the splits.
    '''
    test_frac = 1-train_frac
    pos_range = np.arange(p_end_ind)
    neg_range = np.arange(p_end_ind, data.shape[0])
    np.random.shuffle(pos_range)  # in-place shuffling
    np.random.shuffle(neg_range)
    shuffled_inds = np.append(pos_range, neg_range)
    data = data[shuffled_inds]
    seq_lengths = seq_lengths[shuffled_inds]

    train_pos = data[: int(p_end_ind * train_frac)]
    train_neg = data[p_end_ind: int((train_frac + 0.5 * test_frac) * data.shape[0])]
    test_pos = data[int(p_end_ind * train_frac): p_end_ind]
    test_neg = data[int((train_frac + 0.5 * test_frac) * data.shape[0]):]

    train_pos_l = seq_lengths[:int(p_end_ind * train_frac)]
    train_neg_l = seq_lengths[p_end_ind: int((train_frac + 0.5 * test_frac) * data.shape[0])]
    test_pos_l = seq_lengths[int(p_end_ind * train_frac): p_end_ind]
    test_neg_l = seq_lengths[int((train_frac + 0.5 * test_frac) * data.shape[0]):]

    trainData = np.append(train_pos, train_neg, axis=0)
    testData = np.append(test_pos, test_neg, axis=0)
    train_length = np.append(train_pos_l, train_neg_l)
    test_length = np.append(test_pos_l, test_neg_l)

    train_labs = np.append(np.ones(int(p_end_ind * train_frac), dtype=np.int32),
                           np.zeros(trainData.shape[0] - int(p_end_ind * train_frac), dtype = np.int32))
    test_labs = np.append(np.ones(testData.shape[0]//2, dtype=np.int32),
                          np.zeros(testData.shape[0]//2, dtype =np.int32))

    return trainData, train_length, train_labs, testData, test_length, test_labs


def shuffle(data, seq_lengths, p_end_ind):
    pos_range = np.arange(p_end_ind)
    neg_range = np.arange(p_end_ind, data.shape[0])
    np.random.shuffle(pos_range)  # in-place shuffling
    np.random.shuffle(neg_range)
    shuffled_inds = np.append(pos_range, neg_range)
    data = data[shuffled_inds]
    seq_lengths = seq_lengths[shuffled_inds]
    return data, seq_lengths


# add the labels data for train and get the labels for BATCH!!!!
def getTrainBatch(data, y, seq_lengths, batch_id, batch_size):
    start = batch_id * batch_size
    end = (batch_id + 1) * batch_size
    if end >= data.shape[0]:
        end = data.shape[0]
        batch_id = -1
    batchx = data[start: end]
    batchy = y[start: end]
    batch_seq_length = seq_lengths[start: end]
    return batchx, batchy, batch_seq_length, batch_id


def oh_labels(y, num_classes):
    ohm = np.zeros((len(y), num_classes), dtype=np.int32)
    ohm[np.arange(y.shape[0]), y] = 1
    return ohm


trainD, train_len, train_labs, testD, test_len, test_labs = train_test_split(ids, seq_lengths, 12500, 0.8)

train_p_end_ind = np.where(train_labs==0)[0][0]
print('Index of last positive instance before splitting to train- validation=', train_p_end_ind)
trainD, train_len, train_labs, validD, valid_len, valid_labs = train_test_split(trainD, train_len, train_p_end_ind, 0.8)
train_p_end_ind = np.where(train_labs==0)[0][0]
print('Index of last positive instance AFTER splitting to train- validation=', train_p_end_ind)

train_oh = oh_labels(train_labs, 2)
valid_oh = oh_labels(valid_labs, 2)
test_oh = oh_labels(test_labs, 2)

print('Shape of train_data=', trainD.shape, train_len.shape, train_labs.shape, train_oh.shape)
print('Shape of validationn_data=', validD.shape, valid_len.shape, valid_labs.shape, valid_oh.shape)
print('Shape of test_data=', testD.shape, test_len.shape, test_labs.shape, test_oh.shape)


print('Start index for negative samples=', train_p_end_ind)

x = tf.placeholder(tf.int32, [None, maxSeqLength])
y = tf.placeholder(tf.float32, [None, numClasses])
seq_len = tf.placeholder(tf.int32, [None])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

weights = {
    'w': tf.get_variable(name='w', shape=[lstmUnits, numClasses], initializer=tf.contrib.layers.xavier_initializer())
         }
biases = {
    'b': tf.Variable(tf.constant(0.1, shape=[numClasses]))
        }
'''
Once we have our input data placeholder, we’re going to call the
tf.nn.embedding_lookup() function in order to get our word vectors. 
The call to that function will return a 3-D Tensor of dimensionality batch size
 by max sequence length by word vector dimensions.
'''


# wrapper for dynamic RNN
def dynamicRNN(x, seqlen, weights, biases, keep_prob):
    # x is batch_size * max_seq_length fist we need to get the embeddings
    # now x is [batch_size, max_seq_length, embedding_size]
    x = tf.nn.embedding_lookup(wordVectors, x)
    # prepare data for RNN. Current data shape is (batch_size, seq_max_len, n_input)
    # required data shape is seq_max_len tensors of shape (batch_size, n_input)
    x = tf.unstack(x, maxSeqLength, axis = 1)
    # define LSTM cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)

    # get LSTM cell output, providing 'sequence_length' performs dynamic calculation
    lstm_out, states = tf.contrib.rnn.static_rnn(lstm_cell, x, initial_state = None,
                                                dtype=tf.float32, sequence_length=seqlen)

    # when performing dynamic calculation, we should get the last dynamically computed output,
    # i.e., if a sequence length is 10, we should get the 10th output.
    # 'outputs' is a list of output at every time step: [(batch_size, lstm_units), ..].
    # We pack them in a Tensor and change
    # back dimension to (batch_size, seq_max_len, n_input)
    # note: We did the padding at the end of each sequence!
    lstm_out = tf.stack(lstm_out)
    lstm_out = tf.transpose(lstm_out, [1, 0, 2])
    # Hack to build indexing and get the right output:
    batch_size = tf.shape(lstm_out)[0]

    # end indices for each sample. if we flatten all samples into 20*batch_size
    # we can use this 1D indices to get the right output
    index = tf.range(batch_size) * maxSeqLength + (seqlen - 1)
    print(lstm_out.get_shape())
    # indexing now, outputs contains only the last unit of each sequence)
    print(tf.reshape(lstm_out,[-1, lstmUnits]).get_shape())

    lstm_out = tf.gather(tf.reshape(lstm_out,[-1, lstmUnits]), index, axis=0)
    print(lstm_out.get_shape())
    logits = tf.matmul(lstm_out, weights['w']) + biases['b']
    return logits


logits = dynamicRNN(x, seq_len, weights, biases, keep_prob)
# loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# evaluate model
correct_predictions =  tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
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
    train_summary_writer = tf.summary.FileWriter('tensorboard-vis/train')
    valid_summary_writer = tf.summary.FileWriter('tensorboard-vis/validation')
    test_summary_writer = tf.summary.FileWriter('tensorboard-vis/test')
    patience = 0
    best_valid_accu = 0
    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels, batch_seq_len, batch_id = getTrainBatch(trainD,
                                                                            train_labs,
                                                                            train_len,
                                                                            batch_id,
                                                                            batch_size)
        batch_id += 1
        sess.run(optimizer, {x: nextBatch, y: nextBatchLabels, seq_len: batch_seq_len,
                             learning_rate:init_lr, keep_prob: dropout})

        if patience == lr_decrease_patience:
            init_lr *= lr_decrease_factor

        if patience == early_stop_patience:
            break
        # Write summary to Tensorboard
        if i % 100 == 0:
            train_summary, train_loss, train_accu = sess.run([merged, loss, accuracy],
                                                             feed_dict={x: trainD,
                                                                        y: train_oh,
                                                                        seq_len: train_len,
                                                                        learning_rate: init_lr,
                                                                        keep_prob: 1.})
            valid_summary, valid_loss, valid_accu = sess.run([merged, loss, accuracy],
                                                             feed_dict={x: validD,
                                                                        y: valid_oh,
                                                                        seq_len: valid_len,
                                                                        learning_rate: init_lr,
                                                                        keep_prob: 1.})

            test_summary, test_loss, test_accu = sess.run([merged, loss, accuracy],
                                                          feed_dict={x: testD,
                                                                     y: test_oh,
                                                                     seq_len: test_len,
                                                                     learning_rate: init_lr,
                                                                     keep_prob: 1.})
            train_summary_writer.add_summary(train_summary, i)
            valid_summary_writer.add_summary(valid_summary, i)
            test_summary_writer.add_summary(test_summary,i)
            print('Train data Iteration=', i, 'Loss=', train_loss, 'Accuracy=', train_accu)
            print('valid data, Iteration=', i, 'valid loss=', valid_loss,'Accuracy=', valid_accu)
            print('Test data, Iteration=', i, 'Test loss=', test_loss,'Accuracy=', test_accu)

            if valid_accu > best_valid_accu:  # i just check every 100 batches for now
                patience = 0
                best_valid_accu = valid_accu
                saver.save(sess, "models/pretrained-lstm-%.3f.ckpt" % round(best_valid_accu, 3))
            else:
                patience += 1

    train_summary_writer.close()
    valid_summary_writer.close()
    test_summary_writer.close()
