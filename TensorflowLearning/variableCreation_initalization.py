import tensorflow as tf
#variables must be explicitly intialized and can be saved to disk during and after training.
#you can later restore saved values to exercise or analyze the model
#CREATION:
#when you create a variable you pass a tensor as it's initial value to the Variable() constructor.
#you need to provide shape of tensor as well
#tensorflow provides advanced mechanisms to change the shape the tensor if needed.
#EXAMPLE:
weights=tf.Variable(tf.random_normal([784,200],stddev=0.35),name='weights')
biases=tf.Variable(tf.zeros([200]),name='biases')
####
#DEVICE PLACEMENT:
# A variable can be pinned to a particular device when it is created using a:
#  with tf.device(...) block
#EXAMPLE: Pin a variable to CPU:
#with tf.device("/cpu:0"):
 #   v=tf.Variable(tf.zeros([200]),name='v')
#Pin a Variable to GPU:
#with tf.device("/gpu:0"):
#    v=tf.Variable(tf.zeros([200]),name='v')
#pin a variable to a particular parameter server task:
#with tf.device("/job:ps/task:7"):
#    v=tf.Variable(tf.zeros([200]),name='v')

#device placement is very important when running in a replicated setting.
##########
#INITIALIZING FROM ANOTHER VARIABLE:
#you sometimes need to initialize a variable from the initial value of another variable.
#as the operation(tf.initialize_all_variables() ) initializes all variables
# in parallel you have to be careful when this is needed
#to initialize a new variable from the value of another variable
# use the other variable's initialized_value() property
#EXAMPLE:
weights=tf.Variable(tf.random_normal([784,200],stddev=0.35),name='weights')
#create another variable with the same value as weights:
w2=tf.Variable(weights.initialized_value(),name='w2')
#create another variable with twice the value of the weights:
w_twice=tf.Variable(weights.initialized_value()*2.0,name='w_twice')
#INITIALIZATION:
#variable initializers must be run explicitly before other ops in your model can be run
#the easiest way to do so is to use an operation that runs all variable initializers and
# we need to run that operation before using the model.
#you can alternatuvely restore variable values from a checkpoint file.
init_op=tf.global_variables_initializer()
#later when launching the model:
with tf.Session() as sess:
    sess.run(init_op)
#use the model....
####################
#SAVING AND RESTORING:
#use: tf.train.Saver object.
#the constructor adds 'save' and 'restore' operations to the graph for all or specified list of variables
#in the graph
#the saver object provides methods to run these operations specifying paths for the checkpoint files to write or
#read from.
#CHECK POINT FILES:
#variables are saved in binary files that, roughly, contain a map from variable names to tensor values
#when you create a Save object, you can optionally choose names for the variables
#in the ckeckpoint files.
#by default it uses the value of the Variable.name property for each variable.
#to understand what variables are in a checkpoint, you can use the 'inspect_checkpoint' library
#and in particular , the 'print_tensors_in_checkpoint_file' function.
