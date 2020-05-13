"""
Adversarial examples are specialised inputs created with the purpose of confusing a neural network,
resulting in the misclassification of a given input. These notorious inputs are indistinguishable
to the human eye, but cause the network to fail to identify the contents of the image.
There are several types of such attacks, however, here the focus is on the fast gradient sign method attack,
which is a white box attack whose goal is to ensure mis-classification. A white box attack is
where the attacker has complete access to the model being attacked.

The fast gradient sign method works by using the gradients of the neural network to create an adversarial example.
For an input image, the method uses the gradients of the loss with respect to
the input image to create a new image that maximizes the loss.

A method to accomplish this is to find how much each pixel in the image
contributes to the loss value, and add a perturbation accordingly.
Hence this method

The only goal is to fool an already trained model.
So let's try and fool a pretrained model. In this tutorial, the model is MobileNetV2 model, pretrained on ImageNet.
"""

from __future__ import  absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
# Imagenet Labels
decode_predictions = keras.applications.mobilenet_v2.decode_predictions
print(decode_predictions)


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255.
    image = tf.image.resize(image, (224, 224))
    image = image[None, ...]
    return image


# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
    preds = decode_predictions(probs, top=1)
    print(preds)
    return preds[0][0]


# Original Image
imgurl = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
imgname = 'YellowLabradorLooking_new.jpg'
image_path = tf.keras.utils.get_file(imgname, imgurl)
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)
plt.figure()
plt.imshow(image[0])
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()


# Create Adversarial Image:
loss_object = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        # Since image is not a trainable variable or optimizer so we need to watch it!!
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


# The resulting perturbations can also be visualised.
# Get the input label of the image.
print("Shape of predicted probabilities out of model: {}".format(image_probs.shape))
labrador_retriever_index = 208
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0])
plt.show()


"""
Let's try this out for different values of epsilon and observe the resultant image. 
You'll notice that as the value of epsilon is increased, it becomes easier to fool 
the network. However, this comes as a trade-off which results in the perturbations
 becoming more identifiable.
"""


def display_images(image, description):
    _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
    plt.figure()
    plt.imshow(image[0])
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
    plt.show()


epsilons = [0, 0.01, 0.1, 0.15, 0.2, 0.3]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    display_images(adv_x, descriptions[i])
