import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utils import *
from ops import *
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets('/mnist')
train, validation, test =mnist
trainImages = train.images
print(trainImages.shape)
n_samples = trainImages.shape[0]
n_z =10
batchsize = 200
epochs =100
os.chdir('C:/behrouz/projects2/VAE')

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img
def encoder(input_images):
    with tf.variable_scope("Encoder"):
        h1 = tf.layers.dense(input_images, 100, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu)
        w_mean = tf.layers.dense(h2, n_z, name="w_mean", activation=None,
                                 kernel_constraint=tf.keras.constraints.MaxNorm(5),
                                   bias_constraint=tf.keras.constraints.MaxNorm(1))
        w_stddev = tf.layers.dense(h2,n_z, name="w_stddev", activation=None,
                                   kernel_constraint=tf.keras.constraints.MaxNorm(2),
                                   bias_constraint=tf.keras.constraints.MaxNorm(1))
    return w_mean, w_stddev
def decoder(z):
    with tf.variable_scope("decoder"):
        h1 = tf.layers.dense(z, 100, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu)
        out = tf.layers.dense(h2, 784, activation =tf.nn.sigmoid)
        return out

images = tf.placeholder(tf.float32, [None, 784])
z_mean, z_stddev = encoder(images)
print('Shape of Latent Space mean: ', z_mean.get_shape())
samples = tf.random_normal([batchsize,n_z],0,1,dtype=tf.float32)
guessed_z = z_mean + (tf.square(z_stddev) * samples)
generated_images = decoder(guessed_z)
print('Shape of generated Images:', generated_images.get_shape())
generation_loss = -tf.reduce_sum(images * tf.log(1e-8 + generated_images) +
                                 (1-images) * tf.log(1e-8 + 1 - generated_images),axis =1)
# KL loss between 2 normal distributions:
# -sum(P *log(q/p);  here P is the latent distribution and q is the standard normal distribution.
#KL- divergence between two normal distributions can be derived-> it can be found on google.
# tf.log(tf.square(z_stddev)):
# Since there is no restrivtion on the network to generate positive standard deviation, we square the network output!
KL_loss = 0.5* tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, axis=1)
cost = tf.reduce_mean(generation_loss + KL_loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
def train():
        visualization = mnist.train.next_batch(batchsize)[0]
        reshaped_vis = visualization.reshape(batchsize,28,28)
        print(reshaped_vis.shape)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                for idx in range(int(n_samples / batchsize)):
                    batch = mnist.train.next_batch(batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((optimizer, generation_loss, KL_loss), feed_dict={images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (n_samples - 3) == 0:
                        print ("epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss)))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(generated_images, feed_dict={images: visualization})
                        generated_test = generated_test.reshape(batchsize,28,28)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))

            x_sample, y_sample = mnist.test.next_batch(2000)
            z_mu = sess.run(z_mean, feed_dict={images :x_sample})
            print(z_mu.shape,'===')
            plt.figure(figsize=(8, 6))
            plt.scatter(z_mu[:, 0], z_mu[:, 1], c =y_sample)
            plt.colorbar()
            plt.grid()
            plt.show()


train()