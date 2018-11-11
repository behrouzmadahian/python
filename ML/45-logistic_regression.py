'''
A logistic regression algorithm using Tensorflow.
Used data: MNIST datanse of handwritten digits.
'''

from __future__ import print_function
import tensorflow as tf

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tensorflow graph input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28x28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition (10 classes)

# model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# logistic regression model
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=1)) # this may numerically become unstable.

# instead we'd better use tensorflow built-in function for better numerical stability
# pred = tf.matmul(x, W) + b
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialize variables
init = tf.global_variables_initializer()

# launch default graph
with tf.Session() as session:
	session.run(init)

	# training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.0
		total_batch = int(mnist.train.num_examples / batch_size)

		# loop over all batches
		for i in range(total_batch):
			x_batch, y_batch = mnist.train.next_batch(batch_size)
			# run optimization operation (back propagation) and cost operation (to get cost)
			_, batch_cost = session.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
			# compute average cost
			avg_cost += batch_cost / total_batch

		# display logs every display_step epochs
		if (epoch + 1) % display_step == 0:
			print("Epoch:", (epoch + 1), "cost=", avg_cost)

	print("Optimization finished!")

	# test model on test data
	correct_predicion = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

	# calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_predicion, tf.float32))

	# both of the following approaches are correct for evaluating a tensorflow operation
	#print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
	print("Accuracy:", session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))