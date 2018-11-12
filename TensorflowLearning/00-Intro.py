import numpy as np
import tensorflow as tf
'''
tensorflow is a powerful library for doing large-scale numerical computation.
Tensorflow does most of it's heavy computation outside of python. in order to avoid overhead
of transferring data back to python repeatedly, it waits until we define a graph of
interacting operations that run intirely outside of python.
we describe these interacting operations by manipulating symbolic variables.
the role of the python code is therefore to build this external computation graph, and to dictate
which parts of the computation graph should be run.
Tensorflow relies on a highly efficient C++ backend to do its computation. The connection
to this backend is called a 'session'
we can use tf.session or tf.InteractiveSession
tf.session: the computation graph must be defined completely before starting the session
# tf.InteractiveSession: makes tensorflow more flexible about how you structure your code
# it allows you to interleave operations which build a computation graph with
# ones that run the graph
#this is particularly convenient when working in interactive contexts like IPython
########
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)  # represent each y as a vector of zero/ones
# there are 55k train 10 k test and 5k validation
# labels can be 0-9. each label is represented with a one hot vector as follows
# 3: [0,0,0,1,0,0,0,0,0,0]
print(len(mnist))

train, validation, test = mnist
print(train.images.shape)
print(train.labels.shape)
# it looks like it goes to the training set sequentially and does not choose randomly->good
print(train.next_batch(10)[1])
print(train.next_batch(10)[1])

print(train.labels[0:20])
'''
symbolic variables:
a symbolic  2-d tensor variable of type float32 entries and dimensions: nrow=to be decided by data and 784 columns!
None means the dimension can be of any length!
Symbolic variable representing the input data images:
symbolic variable representing weight matrix:  [784,10]
symbolic variable representing the bias terms  [10]
a Variable is a modifiable tensor that lives in TensorFlow's graph
of interacting operations.
it can be used and even modified by the computation
for machine learning applications, one generally has the model parameters be 'Variables'
we can create these variables by giving tf.Variables the initial value of the Variables
the shape argument in the definition of symbolic variables is optional but it allows
tensorflow to automatically catch bugs stemming from inconsistent tensor shapes
'''

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# defining the output:
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
'''
Training: cost function, what it means for a model to be good!?
one very nice function to determine the loss of a model is called cross-entropy.
cross_entropy=-sum  ( y_i * log(P_i) ) ;  P_i=P(y_i_pred=y_i)
in one hot vector representation. since y_i has only one nonzero element, we can complete multiply the y_i vector
with the predicted probability vector for sample i. ELEMENTWISE!
note: we do element wise multiplication of y and y_pred => choosing column i of the probability vector!!!!
###############
#TRAINING:
###############
#reduce_sum():
Reduces its input along the dimensions given in reduction_indices.
unless keep_dims=True, the rank of the tensor is reduced by one for each entry in reduction_indices
if keep_dims=True, the reduced dimensions are retained with length 1.
if reduction_indices has no entry at all, all dimensions are reduced and a tensor with single element is returned.

tf.reduce_sum adds elements in the second dimension of y, due to the reduction_indices=[1] parameter
finally tf.reduce_mean computes the mean over all examples in the batch
tf.reduce_mean: calculates mean over all examples in the batch
'''


y = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))  # binary true false vector of length of training data
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Since tensorflow knows the entire graph of computations, it can automatically use backpropagation algorithm to
# efficiently determine how your variables effect the loss you ask it to minimize.
# TRAINING:
# learning_rate=0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Tensorflow also provides many other optimization algorithm.
# Initialization:
'''
before performing the training, we need to create an operation to initialize the variables we created.
note that this defines the operation but does not run it yet
we do session.run(init) to initialize the variables in the current session!
'''
init = tf.global_variables_initializer()
# Opening the visualization in pycharm:
# go to View -> Tool Windows ->Terminal ->type :
# tensorboard --logdir=/the location to tensorflow log files
# or in chrome go to:
# http://0.0.0.0:6006

# you can initialize several optimizers at the same time and have few parts in your code that runs each one
# this way the second optimizer picks up from the previous one results and does not go from begining
epochs = 100
total_batches = 100  # I just cam eup with these number to make it run!
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        print('Epoch: %d' % i)
        print('*' * 10)
        for j in range(total_batches):
            batchX, batchY = train.next_batch(100)
            sess.run([train_step], feed_dict={x: batchX, y: batchY})
            print('Training Accuracy: ', sess.run(accuracy, feed_dict={x: train.images, y: train.labels}))
    print(sess.run(accuracy, feed_dict={x: test.images, y: test.labels}))

