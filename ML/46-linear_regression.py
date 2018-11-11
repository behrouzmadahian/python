'''A linear regression algorithm using Tensorflow'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# random generator
rand_gen = np.random

# parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# training data
X_train = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
Y_train = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = X_train.shape[0]

# tensorflow graph input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# model weights
W = tf.Variable(rand_gen.randn(), name="weight")
b = tf.Variable(rand_gen.randn(), name="bias")

# linear model
pred = tf.add(tf.multiply(X, W), b)

# mean square error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

# gradient descent (minimize method learns W and b because they are trainable=True by default)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialize variables
init = tf.global_variables_initializer()

# launch default graph
with tf.Session() as session:
	session.run(init)

	# fit training data
	for epoch in range(training_epochs):
		for (x, y) in zip(X_train, Y_train):
			session.run(optimizer, feed_dict={X: x, Y: y})

		# display logs every display_step epochs
		if (epoch + 1) % display_step == 0:
			c = session.run(cost, feed_dict={X: X_train, Y: Y_train})
			print("Epoch:", (epoch + 1), "cost:", c, "W=", session.run(W), "b=", session.run(b))

	print("Optimization finished!")
	training_cost = session.run(cost, feed_dict={X: X_train, Y: Y_train})
	print("Training cost:", training_cost, "W=", session.run(W), "b=", session.run(b))

	# plot display
	plt.plot(X_train, Y_train, 'ro', label="Original data")
	plt.plot(X_train, session.run(W) * X_train + session.run(b), label="Fitted line")
	plt.legend()
	plt.show()