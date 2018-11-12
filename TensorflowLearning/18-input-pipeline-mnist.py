import tensorflow as tf
import random
import time
'''
A brief example of using tensorflow's input pipeline to load your own custom data structures into tensorflow's 
computational graph. This includes the partitioning of the data into a test and train set and batching 
together a set of images. 
This will not work for large datasets as <ops.convert_to_tensor> will create constants of your data in your graph! 
Only when we start our queue runners right before our session operations the pipeline will be active and loading data.

The input pipeline deals with reading csv files, decode file formats, restructure the data, shuffle the data,
data augmentation or other pre-processing, and load the data in batches using threads.
However, we do have to write some code to get this to work.
 
Note: data is being Read from source raw files. if we convert to TFRecords binary file it will become up to 100 times 
faster!!
'''
# load the label data
data_path = 'C:/behrouz/PythonCodes/ML/TensorflowLearning/mnist/'
test_labels_file = 'test-labels.csv'
train_labels_file = 'train-labels.csv'
NUM_CHANNELS = 1
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
BATCH_SIZE = 512

# The first thing we do is loading the label and image path information from the text files we have generated.


def encode_label(label):
    return int(label)


def read_label_file(file):
    f = open(file, 'r')
    filepaths = []
    labels = []
    for line in f:
        filepath, label = line.split(',')
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels


with tf.device('/cpu:0'):
    train_file_paths, train_labels = read_label_file(data_path + train_labels_file)
    test_file_paths, test_labels = read_label_file(data_path + test_labels_file)
    test_set_size = len(test_file_paths)
    # Do some optional pre-processing on our string lists:
    # for the sake of this example we are also going to concat the given train and test set.
    # We then shuffle the data and create our own train and test set later on.
    # We only do this to show the capabilities of the tensorflow input pipeline!
    all_file_paths = train_file_paths + test_file_paths
    all_labels = train_labels + test_labels
    # Start building the pipeline
    # Make sure that the dtype of our tensor is matching the type of data that we have in our lists.
    from tensorflow.python.framework import ops
    from tensorflow.python.framework import dtypes

    # If you decide to use the train and test data set of mnist as it is given
    # you would just do the next steps for both sets simultaneously.
    # convert string into tensors
    all_images = ops.convert_to_tensor(all_file_paths, dtype=dtypes.string)
    all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)
    # partitioning the data:

    partitions = [0] * len(all_file_paths)
    partitions[:test_set_size] = [1] * test_set_size
    random.shuffle(partitions)
    train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
    train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

    # Build input Queues and define how to load images:
    # slice_input_producer: will slice your tensors into single instances and queue them up using threads
    # There are further parameters to define the number of threads used and the capacity of the queue
    # We then use the path information to read the file into our pipeline and decode it using the jpg decoder
    # (other decoders can be found in the API documentation).
    # we do not want to shuffle within each epoch!
    train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=False)
    test_input_queue = tf.train.slice_input_producer([test_images, test_labels], shuffle=False)
    # process path and string tensor into an image and a label
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    train_label = train_input_queue[1]
    file_content = tf.read_file(test_input_queue[0])
    test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    test_label = test_input_queue[1]
    # group samples into batches
    # If you run train_image in a session you would get a single image i.e. (28, 28, 1)
    # We have yet to start our runners to load images. So far we have only described how the
    # pipeline would look like and tensorflow doesn't know the shape of our images.
    train_image.set_shape([IMAGE_HEIGHT, IMAGE_HEIGHT, NUM_CHANNELS])
    test_image.set_shape([IMAGE_HEIGHT, IMAGE_HEIGHT, NUM_CHANNELS])
    # collect batches of images before processing
    train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label],
                                                                  batch_size=BATCH_SIZE,
                                                                  num_threads=4,
                                                                  capacity=50000,
                                                                  min_after_dequeue=10000,
                                                                  allow_smaller_final_batch=True)
    test_image1, test_label1 = tf.train.batch([test_image, test_label],
                                              batch_size=test_set_size,
                                              num_threads=1, capacity=10000)

# run the Queue runners and start a Session
'''
We have finished building our input pipeline. However, if we would now try to access e.g. test_image_batch,
we would not get any data as we have not started the threads who will load the queues and push data into
our tensorflow objects. After doing that, we will have two loops one going over the training data and 
one going over the test data. You will probably notice that
the loops are bigger than the number of samples in each data set. 

our input pipeline will just cycle over the training data as often as it has to. 
It is your own responsibility to make sure that you correctly count the number of epochs.
'''


def model(x, y, reuse=False):
    with tf.variable_scope('Model', reuse=reuse):
        layer1 = tf.layers.dense(x,
                                 256,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('layer1', layer1.get_shape())
        layer2 = tf.layers.dense(layer1,
                                 128,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('layer2', layer2.get_shape())
        logits = tf.layers.dense(layer2,
                                 10,
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        # binary true false vector of length of training data
        correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return logits, accuracy


# initialize the queue threads to shovel data:
with tf.device('/cpu:0'):
    x = tf.cast(tf.reshape(train_image_batch, [-1, 28 * 28]), tf.float32)
    print('Shape of input=', train_image_batch.get_shape(), x.get_shape())
    y = tf.one_hot(train_label_batch, depth=10)
    logits, accuracy = model(x, y, reuse=False)
    # cost function and optimization
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(cost)
    print(test_image1.get_shape())
    xtest = tf.cast(tf.reshape(test_image1, [-1, 28 * 28]), tf.float32)
    print(xtest.get_shape())
    ytest = tf.one_hot(test_label1, depth=10)
    _, test_accuracy = model(xtest, ytest, reuse=True)

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(200):  # 100 batches
        start_time = time.time()
        loss, acc, _ = sess.run([cost, accuracy, optimizer])
        end_time = time.time()
        if i % 100 == 0:
            print('Batch=', i + 1, 'Loss= ', loss, 'Accuracy= ', acc)
            print('Batch processing time= %.4f(s)' % (end_time - start_time))
    test_accu = sess.run(test_accuracy)
    print('Test Accuracy=', test_accu)

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()


