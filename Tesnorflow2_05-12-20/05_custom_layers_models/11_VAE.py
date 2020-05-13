"""
TODO: 
1. Save and restore the model!!(DONE)
2. bring the whole model out of functional API (DONE)
3. Read the following blog:
https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/ (DONE)
Why encode is made to return variance in log space:
1. log variance can be positive or negative- no need for restriction on model
2. In calculating the KL divergence loss:
In practice, however, it’s better to model SIGMA in log space
 as it is more numerically stable to take exponent compared to computing log
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


class Encoder(keras.layers.Layer):
    """ Maps MNIST digits to a triplet (z_mean, z_log_var, z) """
    def __init__(self, latent_dim=32, intermediate_dim=64, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = keras.layers.Dense(latent_dim)
        self.dense_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()
  
    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
  
  
class Decoder(keras.layers.Layer):
    """ Converts z, the eencoded digit vector, back into a readable digit """
    def __init__(self, original_dim, intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = keras.layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = keras.layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)
  

class VariationalAutoEncoder(keras.Model):
    """ Combines the encoder and decoder into and end-to-end model for training."""
    def __init__(self, original_dim, intermediate_dim,
                 latent_dim, name='vae', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)
      
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        """
        Add KL divergence regularization loss:
        We add KL dvergence between the sampled image in latent space( N(z_mean, exp(0.5 z_log_var))
        and standard normal distribution
        go to above reference  for derivation:
        """
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1.0)
        self.add_loss(kl_loss)
        return reconstructed


original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)
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
# Iterate over epochs:
for epoch in range(epochs):
    print('Start of Epoch {}'.format(epoch))
    # Iterate over the batches of the dataset:
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            if step == 0:
                print('Shape of x_batch: {}'.format(x_batch_train.shape))
                print('Shape of reconstucted image: {}'.format(reconstructed.shape))
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # Add KL divergence regularization
            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_metric(loss)
    reconstructed = vae(test_dataset)
    print('Epoch {}:, Mean loss : {}'.format(epoch, loss_metric.result()))
    print(reconstructed.shape)
    fig = plot(reconstructed.numpy())
    plt.savefig(plot_folder+'_vae_Gen_epoch_{}.png'.format(str(epoch+1)), bbox_inches='tight')
    plt.close(fig)

    
"""
 Since VAE is subclassing model, we could also trained it like this:
"""
keras.backend.clear_session()
del vae
vae = VariationalAutoEncoder(original_dim, 64, 32)
print(vae.summary())
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
