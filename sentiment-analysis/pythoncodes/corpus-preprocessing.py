import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
'''
We will use GloVe pre-trained word embeddings which have length =50
The word2Vec has 3 million words of each 300 length! a little memory intensive!
Glove contains 400000 words with length 50!
The training set we're going to use is the Imdb movie review dataset. 
This set has 25,000 movie reviews, with 12,500 positive reviews and 12,500 negative reviews.
Each of the reviews is stored in a txt file that we need to parse through. 
'''
wordsList = np.load('training_data/wordsList.npy')
print('shape of list of ALL words in corpus=',wordsList.shape)
wordsList =wordsList.tolist()
print('first 10 words in our wordslist=(binary format)',wordsList[:10])
# encode words as UTF-8
wordsList = [word.decode('UTF-8') for word in wordsList]
print('first 10 words in our wordslist=(UTF-8)',wordsList[:10])
#loading word vectors:
wordVectors = np.load('training_data/wordVectors.npy')
print(wordVectors.shape)
print('Loaded the word vectors!')
#lets find the corresponding word vector for say baseball
baseballIndex = wordsList.index('baseball')
print(baseballIndex)
baseball_wordvector = wordVectors[baseballIndex]
print(baseball_wordvector)
# now that we have our vectors, first step is taking an input sentence and then constructing its vector representation.
#assume we have the sentence:
#i thought the movie was incredible and inspiring.
# in order to get the word vectors, we can use tensorflow's embedding lookup function.
#this function take two aruments, one for embedding matrix, the other, the id;s of each of the words.
#the ids vector can be thought of intigerized  representation of the training set.
#this is basically just the row index of each of the words.

'''
The following piece of code will determine total and average number of words in each review.
'''
from os import listdir
from os.path import isfile, join

positive_files = ['training_data/positiveReviews/'+ f for f in listdir('training_data/positiveReviews')
                   if isfile(join('training_data/positiveReviews', f))]

negative_files = ['training_data/negativeReviews/'+ f for f in listdir('training_data/negativeReviews')
                   if isfile(join('training_data/negativeReviews', f))]
numWords = []
for pf in positive_files:
    with open(pf, 'r', encoding='utf-8') as f:

        counter = sum([len(line.split()) for line in f])
        numWords.append(counter)
print('Total number of words in positive texts=', sum(numWords))
for nf in negative_files:
    with open(nf, 'r', encoding='UTF-8') as f:
        counter = sum([len(line.split()) for line in f])
        numWords.append(counter)
print('negative files finished')
numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

#histogram of data
plt.hist(numWords, bins = 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()
'''
From the histogram as well as the average number of words per file, we can safely say 
that most reviews will fall under 250 words, which is the max sequence length value we will set.
'''
maxSeqLength = 250
'''
Lets see how we can take a single file and transform it into our ids matrix. 
This is what one of the reviews looks like in text file format.
'''
fname = positive_files[0]
with open(fname, 'r') as f:
    for line in f:
        print(line)

# now lets convert to an ids matrix
# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

firstFile = np.zeros((maxSeqLength), dtype='int32')
with open(fname) as f:
    indexCounter = 0
    line=f.readline()
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
        try:
            firstFile[indexCounter] = wordsList.index(word)
        except ValueError:
            firstFile[indexCounter] = 399999 #Vector for unknown words
        indexCounter = indexCounter + 1
print(firstFile)
'''
Now, let's do the same for each of our 25,000 reviews.
We'll load in the movie training set and integerize it to get a 25000 x 250 matrix. 
lets save the sequence length as well!
'''
ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
file_lengths = np.zeros(numFiles)
fileCounter = 0
for pf in positive_files:
   curr_file_length = 0
   with open(pf, "r", encoding='UTF-8') as f:
       indexCounter = 0
       for line in f:
           cleanedLine = cleanSentences(line)
           split = cleanedLine.split()
           curr_file_length += len(split)
           for word in split:
               try:
                   ids[fileCounter][indexCounter] = wordsList.index(word)
               except ValueError:
                   ids[fileCounter][indexCounter] = 399999  # Vector for unkown words

               indexCounter = indexCounter + 1

               if indexCounter >= maxSeqLength:
                   break
           if indexCounter >= maxSeqLength:
               break
   file_lengths[fileCounter] = min(maxSeqLength, curr_file_length )
   fileCounter = fileCounter + 1

for nf in negative_files:
   print(nf)
   curr_file_length = 0
   with open(nf, "r", encoding = 'UTF-8') as f:
       indexCounter = 0
       for line in f:
           cleanedLine = cleanSentences(line)
           split = cleanedLine.split()
           curr_file_length += len(split)
           for word in split:
               try:
                   ids[fileCounter][indexCounter] = wordsList.index(word)
               except ValueError:
                   ids[fileCounter][indexCounter] = 399999  # Vector for unknown words

               indexCounter = indexCounter + 1

               if indexCounter >= maxSeqLength:
                   break
           if indexCounter >= maxSeqLength:
               break
   file_lengths[fileCounter] = min(maxSeqLength, curr_file_length )
   fileCounter = fileCounter + 1
#Pass into embedding function and see if it evaluates.
print(ids.shape)
np.save('idsMatrix', ids)
np.save('file_lengths', file_lengths)

'''
Helper Functions:
Below you can find a couple of helper functions that will be useful when training the network in a later step
'''
def getTrainBatch(batch_size):
    labels = []
    arr = np.zeros([batch_size, maxSeqLength])
    for i in range(batch_size):
        if i % 2 == 0:
            num = random.randint(1, 11499)
            labels.append([1, 0])
        else:
            num = random.randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1: num]
    return arr, labels

def getTestBatch(batch_size):
    labels = []
    arr = np.zeros([batch_size, maxSeqLength])
    for i in range(batch_size):
        num = random.randint(11499, 13499)
        if num <= 12499:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]

    return arr, labels

batch_size = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
numDimensions = 50
x = tf.placeholder(tf.float32, [batch_size, maxSeqLength])
y = tf.placeholder(tf.float32, [batch_size, numClasses])
seq_len = tf.placeholder(tf.float32, [batch_size])
drop_out = 0.75
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))


'''
Once we have our input data placeholder, we’re going to call the
tf.nn.lookup() function in order to get our word vectors. 
The call to that function will return a 3-D Tensor of dimensionality batch size
 by max sequence length by word vector dimensions.
'''
data = tf.Variable(tf.zeros([batch_size, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, x)
print(data.get_shape())

#Simple LSTM layer:
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob= drop_out)
# dynamic_rnn: 'rnn_h' is a tensor of shape [batch_size, timestep, cell_state_size]
# 'rnn_c' is a tensor of shape [batch_size, cell_state_size] -> final cell state!

rnn_h, rnn_c = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32, sequence_length= seq_len)
rnn_h = tf.transpose(rnn_h, [1, 0, 2])
last = tf.gather(rnn_h, int(rnn_h.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(loss)
tf.summary.scalar('Train-loss',loss)
tf.summary.scalar('Train-accuracy', accuracy)
merged = tf.summary.merge_all()
sess = tf.InteractiveSession()

train_summary_writer = tf.summary.FileWriter('tensorboard-vis', sess.graph)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch(batch_size)
   sess.run(optimizer, {x: nextBatch, y: nextBatchLabels})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {x: nextBatch, y: nextBatchLabels})
       train_summary_writer.add_summary(summary, i)

   #Save the network every 10,000 training iterations
   if (i % 10000 == 0 and i != 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)
train_summary_writer.close()