"""
Suppose you want to know where an object is located in the image, the shape of that object, which pixel
belongs to which object, etc. In this case you will want to segment the image, i.e., each pixel of the
image is given a label.
the task of image segmentation is to train a neural network to output a pixel-wise mask of the image.
This helps in understanding the image at a much lower level, i.e., the pixel level.
Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging
to name a few.

The dataset that will be used for this tutorial is the Oxford-IIIT Pet Dataset, created by Parkhi et al.
The dataset consists of images, their corresponding labels, and pixel-wise masks. The masks are basically
labels for each pixel. Each pixel is given one of three categories :

Class 1 : Pixel belonging to the pet.
Class 2 : Pixel bordering the pet.
Class 3 : None of the above/ Surrounding pixel.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from IPython.display import clear_output
tfds.disable_progress_bar()

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
print(info)
"""
The following code performs a simple augmentation of flipping an image. In addition, image is normalized to [0,1].
Finally, as mentioned above the pixels in the segmentation mask are labeled either {1, 2, 3}. 
For the sake of convenience, let's subtract 1 from the segmentation mask, resulting in labels 
that are : {0, 1, 2}.
"""


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    if tf.random.uniform(()) > 0.5:
        # flit the image:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


# the data set already contains the required splits of test and train
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


# Visualizing image example and corresponding mask:
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
    display([sample_image, sample_mask])
    break

"""
Defining the model:
The model being used here is a modified U-Net
A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, 
and reduce the number of trainable parameters, a pretrained model can be used as the encoder. 
Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate 
outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow 
Examples in the Pix2pix tutorial.
The reason to output three channels is because there are three possible labels for each pixel. Think of this as
multi-classification where each pixel is being classified into three classes.
"""
OUTPUT_CHANNELS = 3
encoder_model = keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
print('ENCODER model summary: ')
#print(encoder_model.summary())

# Use the activations of these layers
layer_names = [
                'block_1_expand_relu',   # 64x64
                'block_3_expand_relu',   # 32x32
                'block_6_expand_relu',   # 16x16
                'block_13_expand_relu',  # 8x8
                'block_16_project',      # 4x4
              ]
layers = [encoder_model.get_layer(name).output for name in layer_names]
# Feature Extraction model:
down_stack = keras.Model(inputs=encoder_model.inputs, outputs=layers)
down_stack.trainable = False
print('Downstack model summary: -encoder')
print(down_stack.summary())


class InstanceNormalization(tf.keras.layers.Layer):
  """
  Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
  """

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
                                name='scale',
                                shape=input_shape[-1:],
                                initializer=tf.random_normal_initializer(1., 0.02),
                                trainable=True)

    self.offset = self.add_weight(
                                name='offset',
                                shape=input_shape[-1:],
                                initializer='zeros',
                                trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    print('Shape of mean and variance in Instance normalization: axis=[1,2]: {}'.format(mean.shape))
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """
  Upsamples an input.
      Conv2DTranspose => Batchnorm => Dropout => Relu
      Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_dropout: If True, adds the dropout layer
      Returns:
        Upsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


up_stack = [
    upsample(512, 3),  # 4*4 ->    8*8
    upsample(256, 3),  # 8*8 ->    16*16
    upsample(128, 3),  # 16*16 ->  32*32
    upsample(64, 3),   # 32*32 ->  64*64
]

print('-'*300)


def unet_model(output_channels):
    inputs = keras.Input(shape=[128, 128, 3])
    x = inputs
    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections:
    for up, skip in zip(up_stack, skips):
        print('x before up sampling: {}'.format(x.shape))
        x = up(x)
        print('x AFTER up sampling: {}'.format(x.shape))
        concat = keras.layers.Concatenate()
        x = concat([x, skip])
        print('x AFTER concat: {}'.format(x.shape))
        print('-------')

    # this is the last layer of the model
    #  64 * 64 -> 128 * 128
    last = keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')
    x = last(x)
    return keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
print(model.summary())
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
tf.keras.utils.plot_model(model, show_shapes=True, to_file='/Users/behrouzmadahian/Desktop/python/tensorflow2/Unet.png')

# lets try out the model to see what it predicts before training:


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

# lets observe how the model improves while it is training
# to accomplish this, we use following callback:


class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


EPOCHS = 3
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


# make predicitons and display them!!!
show_predictions(test_dataset, 3)
