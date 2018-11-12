'''
Check GPU utility from commandline:
1- add the path of nvidia-smi to your system paths
2- from commandline run:
    nvidia-smi -l 2
'''
# Placing input pipeline operations on the CPU can significantly improve performance
#utilizing the CPU for the input pipeline frees the GPU to focus on training.
# to ensure preprocessing is on the CPU wrap the preprocessing operations as shown below:
'''
with tf.device('/cpu:0'):
    # function to get and process data
'''
#If using tf.estimator.Estimator the input function is automatically placed on the CPU.
###########
#Using tf.data API:
'''
the tf.data API is the recommended API for building input pipelines.
The tf.data API utilizes C++ multi-threading and has a much lower overhead
'''
# Do not use feed_dict!!!
'''
While feeding data using a feed_dict offers a high level of flexibility, in general feed_dict does not provide 
a scalable solution. If only a single GPU is used, the difference between the tf.data API
 and feed_dict performance may be negligible. Our recommendation is to avoid using feed_dict
  for all but trivial examples. In particular, avoid using feed_dict with large inputs:
'''
#Use large files:
'''
Reading large numbers of small files significantly impacts I/O performance. One approach 
to get maximum I/O throughput is to preprocess input data into larger (~100MB) TFRecord files. 
 For smaller data sets (200MB-1GB), the best approach is often to load the entire data set into memory. 
'''
# fused ops:
'''
Fused Ops combine multiple operations into a single kernel for improved performance. 
There are many fused Ops within TensorFlow
Fused batch norm:
Fused batch norm combines the multiple operations needed to do batch normalization into a single kernel. 
Batch norm is an expensive process that for some models makes up a large percentage of the operation time. 
Using fused batch norm can result in a 12%-30% speedup.
bn = tf.layers.batch_normalization(
    input_layer, fused=True, data_format='NCHW')

'''
# RNN performance:
'''
There are many ways to specify an RNN computation in TensorFlow and they have trade-offs 
with respect to model flexibility and performance. 
1- The tf.nn.rnn_cell.BasicLSTMCell should be considered a reference implementation
 and used only as a last resort when no other options will work.
 
When using one of the cells, rather than the fully fused RNN layers, 
you have a choice of whether to use tf.nn.static_rnn or tf.nn.dynamic_rnn. 

There shouldn't generally be a performance difference at runtime, but large unroll amounts can 
increase the graph size of the tf.nn.static_rnn and cause long compile times.
An additional advantage of tf.nn.dynamic_rnn is that it can optionally swap memory 
from the GPU to the CPU to enable training of very long sequences. 


On NVIDIA GPUs, the use of tf.contrib.cudnn_rnn should always be preferred unless you want layer normalization,
 which it doesn't support. It is often at least an order of magnitude faster than tf.contrib.rnn.BasicLSTMCell 
 and tf.contrib.rnn.LSTMBlockCell and uses 3-4x less memory than tf.contrib.rnn.BasicLSTMCell.
 
if you need to run one step of the RNN at a time, as might be the case in reinforcement learning with
a recurrent policy, then you should use the tf.contrib.rnn.LSTMBlockCell with your own environment
interaction loop inside a tf.while_loop construct. 

On CPUs, mobile devices, and if tf.contrib.cudnn_rnn is not available on your GPU, the fastest and most 
memory efficient option is tf.contrib.rnn.LSTMBlockFusedCell
'''