"""
The same as other VAE but the Encoder Decoder architecutres are convolutional
Mnist
Note, it's common practice to avoid using batch normalization when training VAEs,
 since the additional stochasticity due to using mini-batches
 may aggravate instability on top of the stochasticity from sampling.
 loss  is not designed well! will look into this
 
"""
from __future__ import  absolute_import, print_function, division, unicode_literals
import tensorflow as tf
from tensorflow import keras
import time, os, numpy as np, PIL, imageio, glob
from matplotlib import pyplot as plt

(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
train_images /= 255.
test_images /= 255.

# Binarization:
train_images[train_images >= 0.5] = 1.
train_images[train_images < 0.5] = 0.

test_images[test_images >= 0.5] = 1.
test_images[test_images < 0.5] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder_net = tf.keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(latent_dim * 2)  # no activation
        ])
        print(self.encoder_net.summary())
        self.generative_net = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),  # this is Z * N(0, 1) * e^(0.5 * log_z_var)
            keras.layers.Dense(units=7*7*32),
            keras.layers.Reshape(target_shape=(7, 7, 32)),
            keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same', activation='relu')

        ])
        print(self.generative_net.summary())

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(BATCH_SIZE, self.latent_dim))

        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(0.5 * logvar) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


@tf.function
def compute_loss(model, x):
    """
   Note that the MSE loss from before turns to cross entropy here as we binarized x
    """
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    print('Shape of z: {}'.format(z.shape))
    x_logit = model.decode(z)
    print(" Shape of x logits: {}".format(x_logit.shape))

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    print(logpx_z.shape, '=====')
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


epochs = 100
latent_dim = 50
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

plot_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/vae_plot'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(plot_folder + '/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close()


model = CVAE(latent_dim)

generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        compute_apply_gradients(model, train_x, optimizer)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
    generate_and_save_images(
        model, epoch, random_vector_for_generation)
