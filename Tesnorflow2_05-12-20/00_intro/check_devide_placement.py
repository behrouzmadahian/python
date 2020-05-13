"""
If no device placement, tensorflow decides on what device to run an operation- and copies the
tensor to that device - if GPU available -> GPU
"""
import tensorflow as tf
import numpy as np
import time

x = tf.random.uniform([3, 3])
print('Is there a GPU available:')
print(tf.config.experimental.list_logical_devices('GPU'))
print('Is the tensor on GPU:0')
print(x.device.endswith('GPU:0'))
print(x.device)
print('-'*100)
# Explicit device placement:


def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time() - start
    print('10 loops: {:0.3f} ms'.format(1000*result))


# force execution on CPU:
with tf.device('CPU:0'):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith('CPU:0')
    time_matmul(x)

# force execution on GPU #0 if available:
if tf.config.experimental.list_logical_devices('GPU'):
    print('On GPU:')
    with tf.device('GPU:0'):
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith('GPU:0')
        time_matmul(x)

# testing if GPU is available:
print('IS GPU available? :')
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))
