"""
Neural Style Transfer:
Neural style transfer is an optimization technique used to take two images—
a content image and a style reference image (such as an artwork by a famous painter)
—and blend them together so the output image looks like the content image,
but “painted” in the style of the style reference image.

This is implemented by optimizing the output image to:
 a. Match the content statistics of the content image
 b. The style statistics of the style reference image

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
from tensorflow import keras


def tensor_to_image(tensor):
    tensor = tensor * 255.0
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# download images and choose a style image and content image
contentpth =  'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', contentpth)

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


def imshow(image, title=None, show=False):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
  if show:
    plt.show()


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
plt.show()

"""
1- Use the intermediate layers of the model (here VGG19) to get the representations of content and style images
2. try to match the corresponding style representation and target content representation
"""
x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
print("Shape of content image after preprocessing using preprocess_input() of vgg19 :{}".format(x.shape))
x = tf.image.resize(x, (244, 244))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
print('Shape of predictions after passing content image to vgg19: {}'.format(prediction_probabilities.shape))
predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
top_5_key_value = ((class_name, prob) for (number, class_name, prob) in predicted_top_5)
for (number, class_name, prob) in predicted_top_5:
    print('Class name: {:10}, Probability: {:10}'.format(class_name, prob))

# Now load VGG19 without the classification head
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
for layer in vgg.layers:
    print(layer.name)

# Choose intermediate layers from the network to represent the style and content of the image

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# Build model:
# a model that build vgg19 model that returns the intermediate layer outputs:
def vgg_layers(layer_names):
    """Load VGG19 trained on imagenet data"""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = keras.Model([vgg.input], outputs)
    return model


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)
print('-'*50)
print('Style layers statistics: ')
# Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()
print('-'*50)
"""
Calculate Style
Style of an image can be described by means and correlations across different feature maps.
Gram matrix include these!!
"""


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations


# Build a model that returns the style and content tensors
class StyleContentModel(keras.models.Model):
    def __init__(self, style_layers, content_layers):
        """style_layers: list, name of style layers
           content_layers: list, name of content_layers
           When called on an image, this model returns the gram matrix(style) of the style_layers
           and content of the content_layers
        """
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers+content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0, 1]"""
        inputs = inputs * 255.0
        preprocessed_input = keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = outputs[:self.num_style_layers], outputs[self.num_style_layers:]
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content_image))
style_results = results['style']

print('Styles:')
for name, output in sorted(style_results.items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())

"""
run Gradient descent
Do this by calculating the mean square error for your image's output relative 
to each target, then take the weighted sum of these losses.
"""
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

"""
Define a tf.Variable to contain the image to optimize. to make this quick, initialize it with the content image
"""
image = tf.Variable(content_image)
# Since this is a float image, define a function to keep the pixel values between 0 and 1


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


opt = keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
# We use weighted combination of two losses
style_weight = 1e-2
content_weight = 1e4


def style_content_loss(outputs):
    """
    :param outputs: dict with "style" and "content" keys
    output of the call to StyleContentModel
    :return:
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean(tf.square(style_outputs[name] - style_targets[name]))
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean(tf.square(content_outputs[name] - content_targets[name]))
                             for name in content_outputs.keys()])

    content_loss *= content_weight / num_content_layers
    return style_loss + content_loss


@tf.function()
def train_step(image):
    """image is the tensor built by tf.Variable above!"""
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

train_step(image)
train_step(image)
train_step(image)
img_to_display = tensor_to_image(image)
plt.imshow(img_to_display)
plt.show()

# Since it is working, perform a longer optimization:
import time

start = time.time()

epochs = 5
steps_per_epoch = 20

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    # display.clear_output(wait=True)
    # display.display(tensor_to_image(image))
    print("Train step: {}".format(step))
    imshow(image, title='Epoch {}- Step: {}'.format(n+1, step), show=True)


end = time.time()
print("Total time: {:.1f}".format(end - start))

"""
Total variation loss
One downside to this basic implementation is that it produces a lot of high frequency artifacts. 
Decrease these using an explicit regularization term on the high frequency components of the image.
 In style transfer, this is often called the total variation loss:
 
 tf.image.total_variation(image) is the implementation of the two funcs below!
"""


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var


def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


x_deltas, y_deltas = high_pass_x_y(content_image)

plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

plt.subplot(2, 2, 2)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2, 2, 3)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

plt.subplot(2, 2, 4)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")
plt.show()


print('Total variation Loss on content image: ')
print(total_variation_loss(image).numpy())

# We use weighted combination of two losses
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


import time
start = time.time()
# Reinitialize the optimization variable:
image = tf.Variable(content_image)

epochs = 5
steps_per_epoch = 20

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
  # display.clear_output(wait=True)
  # display.display(tensor_to_image(image))
  print("Train step: {}".format(step))
  imshow(image, title='Epoch {}- Step: {}'.format(n + 1, step), show=True)

end = time.time()
print("Total time: {:.1f}".format(end-start))
