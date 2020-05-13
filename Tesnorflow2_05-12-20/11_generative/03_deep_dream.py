"""
This tutorial contains a minimal implementation of DeepDream, as described in this blog post by Alexander Mordvintsev.
DeepDream is an experiment that visualizes the patterns learned by a neural network. Similar to when a child watches
clouds and tries to interpret random shapes, DeepDream over-interprets and enhances the patterns it sees in an image.
It does so by forwarding an image through the network, then calculating the gradient of the image with respect
to the activations of a particular layer. The image is then modified to increase these activations, enhancing
the patterns seen by the network, and resulting in a dream-like image.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import IPython.display as display
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# Choose and image to dream-ifly
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
# Download an image and read it into a NumPy array.


def download(url, max_dim=None):
  name = url.split('/')[-1]
  image_path = tf.keras.utils.get_file(name, origin=url)
  img = PIL.Image.open(image_path)
  if max_dim:
    img.thumbnail((max_dim, max_dim))
  return np.array(img)


# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)


# Display an image
def show(img):
  display.display(PIL.Image.fromarray(np.array(img)))


def imshow(image, title=None, show=False):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
  if show:
    plt.show()


# Downsizing the image makes it easier to work with.
original_img = download(url, max_dim=500)
imshow(original_img, title='Original Image', show=True)

"""
Prepare the feature extraction model
Download and prepare a pre-trained image classification model. You will use InceptionV3 which is similar to 
the model originally used in DeepDream. Note that any pre-trained model will work, 
although you will have to adjust the layer names below if you change this.

The idea in DeepDream is to choose a layer (or layers) and maximize the "loss" in a way that the image 
increasingly "excites" the layers. The complexity of the features incorporated depends on layers chosen 
by you, i.e, 
- lower layers produce strokes or simple patterns, 
- deeper layers give sophisticated features in images, or even whole objects.

The InceptionV3 architecture is quite large (for a graph of the model architecture see TensorFlow's research repo).

For DeepDream, the layers of interest are those where the convolutions are concatenated. 
There are 11 of these layers in InceptionV3, named 'mixed0' though 'mixed10'. Using different 
layers will result in different dream-like images. 
Deeper layers respond to higher-level features (such as eyes and faces), 
while earlier layers respond to simpler features (such as edges, shapes, and textures). 
Feel free to experiment with the layers selected below, but keep in mind that deeper layers 
(those with a higher index) will take longer to train on since the gradient computation is deeper.
"""
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
# Maximize the activations of these layers
names = ['mixed3', 'mixed5']
layers_output = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers_output)
"""
Calculate loss
The loss is the sum of the activations in the chosen layers. The loss is normalized at each layer so 
the contribution from larger layers does not outweigh smaller layers. Normally, loss is a quantity
you wish to minimize via gradient descent. In DeepDream, you will maximize this loss via gradient ascent.
"""


def calc_loss(img, model):
  """
    Pass forward the image through the model to retrieve the activations.
    Calculate the loss as scaled mean of activations at all layers
  """
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return tf.reduce_sum(losses)


"""
Gradient Ascent:
Once you have calculated the loss for the chosen layers, all that is left is to calculate 
the gradients with respect to the image, and add them to the original image.
Adding the gradients to the image enhances the patterns seen by the network.
At each step, you will have created an image that increasingly excites 
the activations of certain layers in the network.
The method that does this, below, is wrapped in a tf.function for performance. 
It uses an input_signature to ensure that the function is not retraced for 
different image sizes or steps/step_size values. See the "Concrete functions guide" for details.
"""


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to 'img'
                # 'GradientTape' only watches 'tf.Variable's by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


deepdream = DeepDream(dream_model)


def run_deep_dream_simple(img, epochs=10, steps_per_epoch=10, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    step = 0
    for epoch in range(epochs):
        step += steps_per_epoch
        loss, img = deepdream(img, steps_per_epoch, tf.constant(step_size))

        #display.clear_output(wait=True)
        # show(deprocess(img))
        # imshow(deprocess(img), title='Deep Dream- epoch {}, step: {}'.format(epoch+1, step), show=True)

        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    # display.clear_output(wait=True)
    imshow(result, title='Final DeepDream Image', show=True)

    return result


# dream_img = run_deep_dream_simple(img=original_img, epochs=10,
#                                   steps_per_epoch=20,  step_size=0.01)

"""
Taking it up an octave
Pretty good, but there are a few issues with this first attempt:

1. The output is noisy (this could be addressed with a tf.image.total_variation loss).
2. The image is low resolution.
3. The patterns appear like they're all happening at the same granularity.

One approach that addresses all these problems is applying gradient ascent at different scales. 
This will allow patterns generated at smaller scales to be incorporated 
into patterns at higher scales and filled in with additional detail.

To do this you can perform the previous gradient ascent approach, 
then increase the size of the image (which is referred to as an octave), 
and repeat this process for multiple octaves.
"""

import time
start = time.time()

OCTAVE_SCALE = 1.30

img = tf.constant(np.array(original_img))
base_shape = tf.shape(img)[:-1]
float_base_shape = tf.cast(base_shape, tf.float32)
print('FLOAT BASE SHAPE: {}'.format(float_base_shape))
for n in range(-2, 3):
  print('n: {} OCTAVE_SCALE*n {}'.format(n, OCTAVE_SCALE**n))
  new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)
  print('New Shape: {}'.format(new_shape))

  img = tf.image.resize(img, new_shape).numpy()
  img = run_deep_dream_simple(img=img, epochs=10, steps_per_epoch=50, step_size=0.01)

# display.clear_output(wait=True)
img = tf.image.resize(img, base_shape)
img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
# show(img)
imshow(img, title='Final IMAGE!', show=True)
end = time.time()
print(end-start)
