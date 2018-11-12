import tensorflow as tf
'''
In tensorflow the supported device types are CPU and GPU.
They are represented as strings. for example:
'/cpu:0' : The cpu on your machine
'/GPU:0' : The GPU on your machine if you have one.
If a tensorflow operation has both GPU and CPU implementations. 
The GPU devices will be given priority when the operation is assigned to a device.
For example, matmul has both CPU and GPU kernels. 
On a system with devices cpu:0 and gpu:0, gpu:0 will be selected to run matmul.

Logging deice placement:
To find out which devices your operations and tensors are assigned to, 
create the session with log_device_placement configuration option set to True.

Manual device palacement:
you can use 'with tf.device' to create a device context such that all operations within
that context will have the same device assignment.

'''
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

#device placement manually:
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

'''
Since a device was not explicitly specified for the MatMul operation(below), the TensorFlow runtime will 
choose one based on the operation and available devices (gpu:0 in this example) 
and automatically copy tensors between devices if required.
This will happen if you install GPU version of tensorflow!
'''
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

'''
Allowing GPU memory Growth:
By default, TensorFlow maps nearly all of the GPU memory of all GPUs visible to the process.
This is done to more efficiently use the relatively precious GPU resources on the devices
by reducing memory fragmentation.
We can control the amount of memory being allocated in two different ways:
1- allow_growth:
attempts to allocate only as much GPU memory based on runtime allocations:
it starts out allocating very little memory, and as Sessions get run and more
GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. 

2-per_process_gpu_memory_fraction:
which determines the fraction of the overall amount of memory that each visible 
GPU should be allocated. For example, you can tell 
TensorFlow to only allocate 40% of the total memory of each GPU 
'''
#1-allow_growth:
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

#2- per_process_gpu_memory_fraction:
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
sess=tf.Session(config=config)
##
'''
If you have more than one GPU in your system, the GPU with the lowest ID 
will be selected by default. If you would like to run on a different GPU,
you will need to specify the preference explicitly
'''