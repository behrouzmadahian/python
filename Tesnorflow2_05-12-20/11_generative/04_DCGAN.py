"""
Deep Convolutional Generative Adversarial Network- DCGAN

During training, the generator progressively becomes better at creating images that look real,
while the discriminator becomes better at telling them apart. The process reaches equilibrium
when the discriminator can no longer distinguish real images from fakes.
"""
from __future__ import  absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
print('Tensorflow Version: {}'.format(tf.__version__))
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
noise_len = 100


def make_generator_model(noise_shape=(100,)):
    inputs = keras.Input(shape=noise_shape, name='noise_gen_input')
    print('Shape and data type of input layer: {}, {}'.format(inputs.shape, inputs.dtype))
    x = keras.layers.Dense(7*7*256, use_bias=False)(inputs)
    x = keras.layers.Flatten()(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Reshape([7, 7, 256])(x)

    x = keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    output = keras.layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2),
                                          padding='same', use_bias=False, activation='tanh')(x)

    model = keras.Model(inputs=inputs, outputs=output, name='Generator')
    return model


def make_discriminator_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)
    logits = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=logits, name='Discriminator_Network')
    return model


generator = make_generator_model(noise_shape=(noise_len,))
print(generator.summary())

# Use untrained model to generate image:
noise = tf.random.normal([1, noise_len])
print(noise.shape)
# Note generator is a keras Model! so we can pass training ARGUMENT for BN to behave differently!!
generated_image = generator(noise)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.title('Randomly Generated Image- Untrained generator')
plt.show()

discriminator = make_discriminator_model()
print(discriminator.summary())
decision = discriminator(generated_image)
print('Probability of randomly generated image being real: {}'.format(tf.sigmoid(decision)[0]))

# Define the loss and optimizers
cross_entropy_loss = keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    """
    :param real_output: P(x_r = real), Output of the Discriminator on real images
    :param fake_output: P(x_g = real), Output of the Discriminator on fake images
    :return: generator loss
    """
    real_loss = cross_entropy_loss(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_loss(tf.zeros_like(fake_output), fake_output)
    return fake_loss + real_loss


def generator_loss(fake_output):
    """
    :param fake_output: P(x_g = real), Output of the Discriminator on fake images
    :return: genrator loss
    """
    return cross_entropy_loss(tf.ones_like(fake_output), fake_output)


# The discriminator and the generator optimizers are different since we will train two networks separately.

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './DCGAN/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
if not os.path.exists(checkpoint_prefix):
    os.makedirs(checkpoint_prefix)
print(checkpoint_prefix)
# defines the stateful objects to checkpoint on demand
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, test_input):
  """
  :param test_input: noise of shape (16, 100)
  Notice 'training' is set to False.
  This is so all layers run in inference mode (batchnorm)
  """
  predictions = model(test_input, training=False)
  fig = plt.figure(figsize=(4, 4))
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('./DCGAN/training_checkpoints/'+'image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()


def train(dataset, epochs):
  for epoch in range(epochs):
    print('Epoch {} begins'.format(epoch+1))
    start = time.time()

    for image_batch in dataset:
      genLoss, discLoss = train_step(image_batch)
    print('Epoch {} final batch Gen Loss: {:7}, Discriminator Loss {:7}'.format(epoch, genLoss, discLoss))
    print('Discriminator loss on Epoch final batch ')

    # Produce images for the GIF as we go
    # display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  #display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, seed)

"""
Note, training GANs can be tricky. It's important that the generator and discriminator 
do not overpower each other (e.g., that they train at a similar rate).

"""
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, EPOCHS)

# restore the latest checkpoint:
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

