'''
A nearest neighbor learning algorithm using Tensorflow.
Nearest Neighbor is a lazy classifier. No work is actually done to train the model. 
It just saves the input points X and all the labels Y. At classification time, 
the predicted class/label is chosen by looking at the nearest neighbor of the input test point.
Used data: MNIST database of handwritten digits.
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# limit the data for this example to 5000 training and 200 training samples
X_train, Y_train = mnist.train.next_batch(5000)
X_test, Y_test = mnist.test.next_batch(200)

# graph inputs
x_train = tf.placeholder("float", [None, 784]) # mnist data image of shape 28x28=784
x_test = tf.placeholder("float", [784]) # image representation of a digit (0-9)

# nearest neighbor calculation using L1 distance
distance = tf.reduce_sum(tf.abs(tf.subtract(x_train, x_test)), axis=1)
# prediction: get minnimum distance index (nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.0

# initializing the variables
init = tf.global_variables_initializer()

# launch the default graph
with tf.Session() as session:
	session.run(init)

	# loop over test data
	for i in range(len(X_test)):
		# get nearest neighbor
		nn_index = session.run(pred, feed_dict={x_train: X_train, x_test: X_test[i, :]})
		# get nearest neighbor class label and compare it with its true label
		print("Test", i, "Prediction:", np.argmax(Y_train[nn_index]), "True class:", np.argmax(Y_test[i]))
		# calculate accuracy
		if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
			accuracy += 1.0/len(X_test)
	print("Done!")
	print("Accuracy:", accuracy)