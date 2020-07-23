"""
Early interpretability methods for neural networks assigned feature importance scores using gradients,
which tell you which pixels have the steepest local relative to your model's prediction at a given
point along your model's prediction function. However, gradients only describe local changes in
your model's prediction function with respect to pixel values and do not fully describe your
entire model prediction function. As your model fully "learns" the relationship between
the range of an individual pixel and the correct ImageNet class, the gradient for this
pixel will saturate, meaning become increasingly small and even go to zero.
Consider the simple model function below:
"""

"""
PLOTS:

left: Your model's gradients for pixel x are positive between 0.0 and 0.8 but go to 0.0 between 0.8 and 1.0. 
Pixel x clearly has a significant impact on pushing your model toward 80% predicted probability on the true class. 
Does it make sense that pixel x's importance is small or discontinuous?

right: The intuition behind IG is to accumulate pixel x's local gradients and attribute its importance 
as a score for how much it adds or subtracts to your model's overall output class probability. 
You can break down and compute IG in 3 parts:

1. interpolate small steps along a straight line in the feature space between 0 
(a baseline or starting point) and 1 (input pixel's value)

2. compute gradients at each step between your model's predictions with respect to each step

3. approximate the integral between your baseline and input by accumulating (cumulative average) these local gradients.

To reinforce this intuition, you will walk through these 3 parts by applying IG to the example "Fireboat" image below.

"""
import tensorflow as tf
from matplotlib import pyplot as plt


def f(x):
    """A Simplified model function"""
    return tf.where(x < 0.8, x, 0.8)


def interpolated_path(x):
    """A straight line path"""
    return tf.zeros_like(x)


x = tf.linspace(start=0.0, stop=1.0, num=6)
y = f(x)

fig = plt.figure(figsize=(12, 5))
ax0 = fig.add_subplot(121)
ax0.plot(x, f(x), marker='o')
ax0.set_title('Gradients saturate over F(x)', fontweight='bold')
ax0.text(0.2, 0.5, 'Gradients > 0 = \n x is important')
ax0.text(0.7, 0.85, 'Gradients = 0 \n x not important')
ax0.set_yticks(tf.range(0, 1.5, 0.5))
ax0.set_xticks(tf.range(0, 1.5, 0.5))
ax0.set_ylabel('F(x) - model true class predicted probability')
ax0.set_xlabel('x - (pixel value)')

ax1 = fig.add_subplot(122)
ax1.plot(x, f(x), marker='o')
ax1.plot(x, interpolated_path(x), marker='>')
ax1.set_title('IG intuition', fontweight='bold')
ax1.text(0.25, 0.1, 'Accumulate gradients along path')
ax1.set_ylabel('F(x) - model true class predicted probability')
ax1.set_xlabel('x - (pixel value)')
ax1.set_yticks(tf.range(0, 1.5, 0.5))
ax1.set_xticks(tf.range(0, 1.5, 0.5))
ax1.annotate('Baseline', xy=(0.0, 0.0), xytext=(0.0, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.1))
ax1.annotate('Input', xy=(1.0, 0.0), xytext=(0.95, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.1))
plt.show()

"""
Establish a baseline:
A baseline is an input image used as a starting point for calculating feature importance. 
Intuitively, you can think of the baseline's explanatory role as representing the impact of the absence 
of each pixel on the "Fireboat" prediction to contrast with its impact of each pixel on the "Fireboat" prediction 
when present in the input image. 
As a result, the choice of the baseline plays a central role in interpreting and visualizing pixel feature importances.

For additional discussion of baseline selection, see the resources in the "Next steps" 
section at the bottom of this tutorial. Here, you will use a black image whose pixel values are all zero.

Other choices you could experiment with include an all white image, or a random image, 
which you can create with tf.random.uniform(shape=(224,224,3), minval=0.0, maxval=1.0).
"""
baseline = tf.zeros(shape=(224, 224, 3))
plt.imshow(baseline)
plt.title('Baseline')
plt.axis('off')
plt.show()

