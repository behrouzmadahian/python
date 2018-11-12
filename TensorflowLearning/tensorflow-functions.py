import tensorflow as tf
'''
1- tf.shape(tensor): 
returns the dynamic shape of the tensor -> its calculated when it is run in session and values of
feed dict provided.

2- tensor.get_shape(): returns the static shape of the tensor which is fixed.
if the shape of the tensor is not known and we would like to use this shape
to do some computation in graph, DO NOT use this!! example: dynamicRNN line 121!

'''