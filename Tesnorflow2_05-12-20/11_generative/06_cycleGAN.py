"""
Unpaired image to image translation using conditional GANs

The paper proposes a method that can capture the characteristics of one image domain and
figure out how these characteristics could be translated into another image domain, all in absence of any paired
training examples.

CycleGAN uses a cycle consistency loss to enable training without the need for paired data.
In other words, it can translate from one domain to another without a one-to-one
mapping between the source and target domain.
This tutorial trains a model to translate from images of horses, to images of zebras.

You can find this dataset and similar ones:
https://www.tensorflow.org/datasets/catalog/overview#cycle_gan

For Discriminator Loss on generated samples, uses a buffer of last 50  generated samples
rather that immediately generated one - helps with Oscilation
replaces log likelihoods with least squares! helps with stability: Least Square GANS

We have two GAN setup:
1. G(x), D(y)
G loss: (D(G(x)) - 1)^2
D loss: (D(y) - 1)^2 + (D(G(x))^2

2. F(y), D(x)
F loss: (D(F(x)) - 1)^2
D1 loss: (D1(x) - 1)^2 + (D1(F(y))^2

forward cycle consistency  loss:
Lc = |F(G(x)) - 1| Norm 1

Backward cycle consistency loss:
Lc = |G(F(y)) - 1| Norm 1

It is like two Auto encoders with encoder decoder architecure

That maps a sample to itself via an intermediate representation that is a
translation of the image into another domain
that in the hidden layer learn the data distribution
of the intermediate data. E.g x -> y (learns this in the process!) -> x
and vice verasa

Later on check to see WHY map function requires label tf.dataset
Cyclegan uses instance normalization instead of batch normalization.

The CycleGAN paper uses a modified resnet based generator.
This tutorial is using a modified unet generator for simplicity

In cycle consistency loss: (forward cycle consistency)

- Image x is passed via generator G that yields generated image Y_pred.
- Generated image y_pred is passed via generator F  that yields cycled image x_pred.
- Mean absolute error is calculated between x_pred and x.

Identity loss:
They find that is is helpful to introduce an additional loss to encourage the generator to be
close to identity mapping.
it is helpful when the real samples of target domain are provided as input to generator!

Generator  G is responsible for translating image X to image Y.
Identity loss says that, if you fed image  Y to generator G,
it should yield the real image Y or something close to image Y
identity loss:
|G(Y) - Y| + |F(x) - X|

In other words, G translates X to Y
Now if you feed it Y it should understand that Y is what it needs to generate and as such
return Y or something close to it!


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
import matplotlib.pyplot as plt

dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image


train_horses = train_horses.map(
    preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

train_zebras = train_zebras.map(
    preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_horse[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Horse with random jitter')
plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)
plt.show()

OUTPUT_CHANNELS = 3

generator_h_z = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_z_h = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_isHorse = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_isZebra = pix2pix.discriminator(norm_type='instancenorm', target=False)

to_zebra = generator_h_z(sample_horse)
to_horse = generator_z_h(sample_zebra)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for i in range(len(imgs)):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    if i % 2 == 0:
      plt.imshow(imgs[i][0] * 0.5 + 0.5)
    else:
      plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real zebra?')
plt.imshow(discriminator_isZebra(sample_zebra)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_isHorse(sample_horse)[0, ..., -1], cmap='RdBu_r')
plt.colorbar()
plt.show()

LAMBDA = 10

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    """
    :param real: Logit of pathces being real  k*k - here 30*30
    :param generated: generated image by associated generator
    :return: discriminator loss
    Note that paper uses least square loss!
    """
    real_loss = loss_object(tf.ones_like(real), real)
    generated_loss = loss_object(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_object(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image-cycled_image))
    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    """
    :param real_image: Y
    :param same_image: G(Y)
    :return:
    """
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


generator_hz_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_zh_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_isHorse_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_isZebra_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./cycle_GAN_checkpoints/train"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

ckpt = tf.train.Checkpoint(generator_g=generator_h_z,
                           generator_f=generator_z_h,
                           discriminator_x=discriminator_isHorse,
                           discriminator_y=discriminator_isZebra,
                           generator_g_optimizer=generator_hz_optimizer,
                           generator_f_optimizer=generator_zh_optimizer,
                           discriminator_x_optimizer=discriminator_isHorse_optimizer,
                           discriminator_y_optimizer=discriminator_isZebra_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


def generate_images(model, test_input, epoch):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(checkpoint_path + '/image_at_epoch_{:04d}.png'.format(epoch))

    # plt.show()

@tf.function
def train_step(real_horse, real_zebra):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_zebra = generator_h_z(real_horse, training=True)
        cycled_horse = generator_z_h(fake_zebra, training=True)

        fake_horse = generator_z_h(real_zebra, training=True)
        cycled_zebra = generator_h_z(fake_horse, training=True)

        # same_x and same_y are used for identity loss.
        same_zebra = generator_h_z(real_zebra, training=True)
        same_horse = generator_z_h(real_horse, training=True)

        disc_real_horse = discriminator_isHorse(real_horse, training=True)
        disc_real_zebra = discriminator_isZebra(real_zebra, training=True)

        disc_fake_horse = discriminator_isHorse(fake_horse, training=True)
        disc_fake_zebra = discriminator_isZebra(fake_zebra, training=True)

        # calculate the loss
        gen_hz_loss = generator_loss(disc_fake_zebra)
        gen_zh_loss = generator_loss(disc_fake_horse)

        total_cycle_loss = calc_cycle_loss(real_horse, cycled_horse) + calc_cycle_loss(real_zebra, cycled_zebra)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_hz_loss = gen_hz_loss + total_cycle_loss + identity_loss(real_zebra, same_zebra)
        total_gen_zh_loss = gen_zh_loss + total_cycle_loss + identity_loss(real_horse, same_horse)

        disc_zh_loss = discriminator_loss(disc_real_horse, disc_fake_horse)
        disc_hz_loss = discriminator_loss(disc_real_zebra, disc_fake_zebra)

    # Calculate the gradients for generator and discriminator
    generator_hz_gradients = tape.gradient(total_gen_hz_loss,
                                           generator_h_z.trainable_variables)
    generator_zh_gradients = tape.gradient(total_gen_zh_loss,
                                           generator_z_h.trainable_variables)

    discriminator_isZebra_gradients = tape.gradient(disc_hz_loss,
                                                    discriminator_isZebra.trainable_variables)
    discriminator_isHorse_gradients = tape.gradient(disc_zh_loss,
                                                    discriminator_isHorse.trainable_variables)

    # Apply the gradients to the optimizer
    generator_hz_optimizer.apply_gradients(zip(generator_hz_gradients,
                                              generator_h_z.trainable_variables))

    generator_zh_optimizer.apply_gradients(zip(generator_zh_gradients,
                                              generator_z_h.trainable_variables))

    discriminator_isZebra_optimizer.apply_gradients(zip(discriminator_isZebra_gradients,
                                                        discriminator_isZebra.trainable_variables))

    discriminator_isHorse_optimizer.apply_gradients(zip(discriminator_isHorse_gradients,
                                                    discriminator_isHorse.trainable_variables))


EPOCHS = 20
for epoch in range(EPOCHS):
    start = time.time()
    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        train_step(image_x, image_y)
        if n % 10 == 0:
          print('.', end='')
    n += 1

    # clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    generate_images(generator_h_z, sample_horse, epoch+1)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
