"""
This tutorial demonstrates how to implement Integrated Gradients (IG), an Explainable AI technique introduced
in the paper Axiomatic Attribution for Deep Networks.
IG aims to explain the relationship between a model's predictions in terms of its features.
It has many use cases including understanding feature importances, identifying data skew,
and debugging model performance.

IG has become a popular interpretability technique due to its broad applicability to any differentiable model
(e.g. images, text, structured data), ease of implementation,
theoretical justifications, and computational efficiency relative to alternative approaches
that allows it to scale to large networks and feature spaces such as images.
"""
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

"""
hub.KerasLayer():
wraps a SavedModel as a keras layer

"""
# Download a pretrained image classifier from TF-hub: Inception V1
model = tf.keras.Sequential([hub.KerasLayer(name='inception_v1',
                                            handle='https://tfhub.dev/google/imagenet/inception_v1/classification/4',
                                            trainable=False)])
"""
build(Input shape)
creates the variables of the layer (optional, for subclass implementers)
"""
# output is a tensorf of (batch_size, 1001)
model.build([None, 224, 224, 3])
model.summary()


def load_imagenet_labels(file_path):
    label_file = tf.keras.utils.get_file('ImageNetLabels.txt', file_path)
    with open(label_file) as reader:
        f = reader.read()
        labels = f.splitlines()
    return np.array(labels)


def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image


imagenet_labels = load_imagenet_labels('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')



img_url = {
    'Fireboat': 'http://storage.googleapis.com/download.tensorflow.org/example_images/San_Francisco_fireboat_showing_off.jpg',
    'Giant Panda': 'http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg',
}

img_paths = {name: tf.keras.utils.get_file(name, url) for (name, url) in img_url.items()}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}

plt.figure(figsize=(8, 8))
for n, (name, img_tensors) in enumerate(img_name_tensors.items()):
    ax = plt.subplot(1, 2, n+1)
    ax.imshow(img_tensors)
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.show()
######
# classify images:


def top_k_predictions(img, k=3):
    image_batch = tf.expand_dims(img, 0)
    predictions = model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    top_labels = imagenet_labels[tuple(top_idxs)]
    return top_labels, top_probs[0]


for(name, img_tensor) in img_name_tensors.items():
    plt.imshow(img_tensor)
    plt.title(name, fontweight='bold')
    plt.axis('off')
    plt.show()
    pred_label, pred_prob = top_k_predictions(img_tensor)
    for label, prob in zip(pred_label, pred_prob):
        print(f'{label}: {prob:0.1%}')
        print('======')

######
# Calculate Integrated Gradients:
# 1. generate a linear interpolation between the baseline and the original image.
# you can think of interpolated images as small steps in the feature space between your baseline and input
# represented by alpha in the original equation
baseline = tf.zeros(shape=(224, 224, 3))
plt.imshow(baseline)
plt.title('Baseline')
plt.axis('off')
plt.show()
alphas = tf.linspace(start=0.0, stop=1.0, num=51)


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


# generate interpolated images along a linear path at alpha intervals
# between black baseline image and the fireboat image
interpolated_images = interpolate_images(baseline=baseline, image=img_name_tensors['Fireboat'], alphas=alphas)

# visualizing the interpolated images
# another way to think about the alpha constant is that it is consistently increasing
# each interpolated image's intensity
fig = plt.figure(figsize=(20, 20))
i = 0
print(alphas[0::10])
print(alphas[::10])
for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
    i += 1
    plt.subplot(1, len(alphas[0::10]), i)
    plt.title(f'alpha: {alpha:.1f}')
    plt.imshow(image)
    plt.axis('off')
plt.tight_layout()
plt.show()


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


# compute the gradients for each image along the interpolation path with respect to the correct ouput

path_gradients = compute_gradients(images=interpolated_images, target_class_idx=tf.constant(555))

# note the output shape of ( n_interpolated_images, img_height, img_width, RGB)
# which gives us the gradient for every pixel of every image along the interpolation path
# thin of these gradients as measuring the change in your
# model's predictions for each small step in the feature space
print(path_gradients.shape)

# recall that the gradients calculated above describe local changes
# to your model's predicted probability of Fireboat and can saturate
# These concepts are visualized using the gradients you calculated above in the two plots below
pred = model(interpolated_images)
pred_proba = tf.nn.softmax(pred, axis=-1)[:, 555]
plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])

ax2 = plt.subplot(1, 2, 2)
# Average across interpolation steps
average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
# Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1])
plt.show()

# How to accumulate these gradients to accurately approximate how each pixel impacts your fireboat predicted prob

# Accumulate Gradients(integral approximation):


def integral_approximation(gradients):
    # Riemann_Trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


ig = integral_approximation(path_gradients)
print('Integrated Gradient Shape: {}'.format(ig.shape))

"""
Now you will combine the 3 previous general parts together into an IntegratedGradients function
 and utilize a @tf.function decorator to compile it into a high performance callable Tensorflow graph. 
 This is implemented as 5 smaller steps below
"""
@tf.function
def integrated_gradients(baseline, image, target_class_idx,
                         m_steps=300, batch_size=32):
  # 1. Generate alphas
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps)

  # Accumulate gradients across batches
  integrated_gradients = tf.zeros_like(image)

  # Batch alpha images
  ds = tf.data.Dataset.from_tensor_slices(alphas).batch(batch_size)

  for batch in ds:
    # 2. Generate interpolated images
    batch_interpolated_inputs = interpolate_images(baseline=baseline, image=image, alphas=batch)

    # 3. Compute gradients between model outputs and interpolated inputs
    batch_gradients = compute_gradients(images=batch_interpolated_inputs,
                                        target_class_idx=target_class_idx)
    print('Batch Gradients: {}'.format(batch_gradients))
    # 4. Average integral approximation. Summing integrated gradients across batches.
    intg_approx = integral_approximation(gradients=batch_gradients)
    print('Integral approximation on the batch: {}'.format(intg_approx))

    integrated_gradients += intg_approx

  # 5. Scale integrated gradients with respect to input
  print('Integrated Gradient Shape: {}'.format(integrated_gradients.shape))

  scaled_integrated_gradients = (image - baseline) * integrated_gradients
  return scaled_integrated_gradients


ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['Fireboat'],
                                       target_class_idx=555)
print('Shape of Integrated Gradients attributes: {}'.format(ig_attributions.shape))

def plot_img_attributions(baseline,
                          image,
                          target_class_idx,
                          m_steps=tf.constant(50),
                          cmap=None,
                          overlay_alpha=0.4):

  attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      m_steps=m_steps)

  # Sum of the attributions across color channels for visualization.
  # The attribution mask shape is a grayscale image with height and width
  # equal to the original image.
  attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

  fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

  axs[0, 0].set_title('Baseline image')
  axs[0, 0].imshow(baseline)
  axs[0, 0].axis('off')

  axs[0, 1].set_title('Original image')
  axs[0, 1].imshow(image)
  axs[0, 1].axis('off')

  axs[1, 0].set_title('Attribution mask')
  axs[1, 0].imshow(attribution_mask, cmap=cmap)
  axs[1, 0].axis('off')

  axs[1, 1].set_title('Overlay')
  axs[1, 1].imshow(attribution_mask, cmap=cmap)
  axs[1, 1].imshow(image, alpha=overlay_alpha)
  axs[1, 1].axis('off')

  plt.tight_layout()
  plt.show()
  return fig


# Looking at the attributions on the "Fireboat" image, we can see the model identifies
# the water cannons and spouts as contributing to its correct prediction.
_ = plot_img_attributions(image=img_name_tensors['Fireboat'],
                          baseline=baseline,
                          target_class_idx=555,
                          m_steps=2400,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.4)

_ = plot_img_attributions(image=img_name_tensors['Giant Panda'],
                          baseline=baseline,
                          target_class_idx=555,
                          m_steps=2400,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.4)

"""
Limitations:
1. Integrated Gradients provides feature importances on individual examples, however, 
it does not provide global feature importances across an entire dataset.

2. Integrated Gradients provides individual feature importances, but it does not explain 
feature interactions and combinations.
"""
