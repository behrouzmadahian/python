"""
The same as prev model but not object oriented!!!
TODO:

1. Save and restore the model!!
2. bring the whole model out of functional API
3. Read the following blog:
https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
"""

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(samples.shape[0]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(samples[i].reshape(28, 28), cmap='Greys_r')
    return fig


class Sampling(keras.layers.Layer):
    """
    Uses (Z_mean, Z_log_var) to sample z, the vector encoding digit
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def encoder(input, latent_dim, intermediate_dim):
    dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')(input)
    z_mean = keras.layers.Dense(latent_dim, activation='relu')(dense_proj)
    z_log_var = keras.layers.Dense(latent_dim, activation='relu')(dense_proj)
    model = keras.Model(inputs=input, outputs=[z_mean, z_log_var])
    return model


def decoder(latent_dim, original_dim, intermediate_dim=64):
    z = keras.Input(shape=(latent_dim,), name='zs')
    dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')(z)
    dense_output = keras.layers.Dense(original_dim, activation='sigmoid')(dense_proj)
    model = keras.Model(inputs=z, outputs=dense_output)
    return model


class Vae_kld(keras.layers.Layer):
    """Gets the layer add the loss to loss in the graph and returns the layer!
    KL divergence between N(mu, SIGMA) and N(0, 1)!!"""

    def call(self, inputs, scale=1.0):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1.0)
        self.add_loss(tf.reduce_sum(kl_loss * scale))
        return inputs


intermediate_dim = 64
latent_dim = 32
original_dim = 784
vae_input = keras.Input(shape=(784, ), name='digits')
encoder_model = encoder(vae_input, latent_dim, intermediate_dim)
z_mean, z_log_var = encoder_model(vae_input)
z_mean, z_log_var = Vae_kld()((z_mean, z_log_var), scale=1e-2)
sampler = Sampling()
sampled = sampler((z_mean, z_log_var))
decoder_model = decoder(latent_dim, original_dim, intermediate_dim)
generated_img = decoder_model(sampled)
vae = keras.Model(inputs=vae_input, outputs=generated_img)

original_dim = 784
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.
x_test = x_test.reshape(10000, 784).astype('float32') / 255.

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
test_dataset = x_test[:32]

plot_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/vae_plot'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
epochs = 30
plot_batches = 1
# Iterate over epochs:
for epoch in range(epochs):
    print('Start of Epoch {}'.format(epoch))
    # Iterate over the batches of the dataset:
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # Add KL divergence regularization
            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_metric(loss)
    reconstructed = vae(test_dataset)
    print('Epoch {}:, Mean loss : {}'.format(epoch, loss_metric.result()))
    print("Shape of reconstructed test data: {}".format(reconstructed.shape))
    fig = plot(reconstructed.numpy())
    plt.savefig(plot_folder + '_vae_Gen_epoch_{}.png'.format(str(epoch + 1)), bbox_inches='tight')
    plt.close(fig)


keras.backend.clear_session()
del vae
vae = keras.Model(inputs=vae_input, outputs=generated_img)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
