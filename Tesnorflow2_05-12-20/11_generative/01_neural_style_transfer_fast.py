"""
Neural Style Transfer:
Neural style transfer is an optimization technique used to take two images—
a content image and a style reference image (such as an artwork by a famous painter)
—and blend them together so the output image looks like the content image,
but “painted” in the style of the style reference image.

This is implemented by optimizing the output image to match the content statistics
of the content image and the style statistics of the style reference image.

This tutorial uses deep learning to compose one image in the style of another image
(ever wish you could paint like Picasso or Van Gogh?).
This is known as neural style transfer and the technique is outlined in
A Neural Algorithm of Artistic Style (Gatys et al.).

This tutorial demonstrates the original style-transfer algorithm. It optimizes the image
content to a particular style. Modern approaches train a model to generate the stylized
image directly (similar to cyclegan). This approach is much faster (up to 1000x).
A pretrained Arbitrary Image Stylization module is available in TensorFlow Hub,
and for TensorFlow Lite.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub

def tensor_to_image(tensor):
    tensor = tensor * 255.0
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# download images and choose a style image and content image
contnetpth =  'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', contnetpth)

style_pth = 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg', style_pth)

# Visualize the input:
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
plt.show()

# Fast style transfer using TF-Hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
stylized_image = tensor_to_image(stylized_image)
plt.imshow(stylized_image)
plt.show()

