import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
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

# now that we have our vectors, first step is taking an input sentence and then constructing its vector representation.
#assume we have the sentence:
#i thought the movie was incredible and inspiring.
# in order to get the word vectors, we can use tensorflow's embedding lookup function.
#this function take two arguments, one for embedding matrix, the other, the id;s of each of the words.
#the ids vector can be thought of intigerized  representation of the training set.
#this is basically just the row index of each of the words.

'''
max- seq length must be determined in the preprocessing file!
'''
vocab_size = 111380 + 1
vec_length = 100
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


'''
Helper Functions:
Below you can find a couple of helper functions that will be useful when training the network in a later step
'''
vocab_list = list(np.loadtxt('vocabulary_list-trainingEmbedding.txt', delimiter='\t', dtype='str'))

weights ={
    'wordVecs':tf.get_variable(name='wordVecs', shape=[vocab_size, vec_length],
                               initializer=tf.contrib.layers.xavier_initializer()),
    'w': tf.get_variable(name='w', shape=[lstmUnits, numClasses], initializer=tf.contrib.layers.xavier_initializer())
         }
biases ={
    'b': tf.Variable(tf.constant(0.1, shape=[numClasses]))
        }
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'training-embedding/modelckpt/lstm-0.874.ckpt')

# link the wordvecs tensor to its metadata file (e.g labels)
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = weights['wordVecs'].name

with open('training-embedding/modelckpt/embd/metadata.tsv', 'w') as f:
    for word in vocab_list:
        f.write(word+'\n')


embedding.metadata_path = 'training-embedding/modelckpt/embd/metadata.tsv'

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter('training-embedding/modelckpt/embd', sess.graph)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)


summary_writer.close()