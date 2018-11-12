import tensorflow as tf
import time

'''
GPUs and TPUs can radically reduce the time required to execute a single training step. 
Achieving peak performance requires an efficient input pipeline that delivers data for
the next step before the current step has finished.

The tf.data API helps to build flexible and efficient input pipelines. 

We need to use queue system of tf.
You can think about it as designing your input pipeline before hand right into the graph and stop doing everything 
in Python! In fact, we will try to remove any Python dependency we have from the input pipeline.
This will also give us nice properties of multi-threading, asynchronicity and memory
optimisation due to the removal of the feed_dict system

There are three main methods of getting data into a TensorFlow program:

1. Feeding: Python code provides the data when running each step.-> slow!!!
2. Reading from files: an input pipeline reads the data from files at the beginning of a TensorFlow graph.
3. Pre loaded data: a constant or variable in the TensorFlow graph holds all the data (for small data sets).

A typical pipeline for reading records from files has the following stages:

1-The list of file names
2-Optional filename shuffling
3-Optional epoch limit
3-Filename queue
4-A Reader for the file format
5-A decoder for a record read by the reader
6-Optional pre processing
8-Example queue
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

train, validation, test = mnist
trainX, trainY = train.images, train.labels
print(trainX.shape)
