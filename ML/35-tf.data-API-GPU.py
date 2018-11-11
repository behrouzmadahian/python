import tensorflow as tf
import random, os, time
import argparse, sys
'''
We converted our data into TFREcords for train, validation, and test data.
Now we want to design the pipeline to read these data!!
If we create a Dataset using a placeholder for data, we can dynamically change the  data
inside the Data set -> this is helpful when we have Validation and Train and we want to switch between
the two and both are say in TFRecords format and we create data set from each!
On a typical system, there are multiple computing devices. In TensorFlow, the supported device types are CPU and GPU. They are represented as strings. For example:

"/cpu:0": The CPU of your machine.
"/device:GPU:0": The GPU of your machine, if you have one.
"/device:GPU:1": The second GPU of your machine, etc.
If a TensorFlow operation has both CPU and GPU implementations, 
the GPU devices will be given priority when the operation is assigned to a device. 
To find out which devices your operations and tensors are assigned to, create the 
session with log_device_placement configuration option set to True.
Manual Device Placement:
If you would like a particular operation to run on a device of your choice instead
 of what's automatically selected for you, you can use with tf.device
Add allow_soft_placement = True so that you can optionally assign different operations to different devices.

Allowing GPU memory growth:
 grow the memory usage as is needed by the process.
 allow_growth option:
 , which attempts to allocate only as much GPU memory based on runtime allocations: 
 it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed,
  we extend the GPU memory region needed by the TensorFlow process. 
 config = tf.ConfigProto()
config.gpu_options.allow_growth = True

per_process_gpu_memory_fraction option:
 which determines the fraction of the overall amount of memory that each visible GPU should be allocated.
  For example, you can tell TensorFlow to only allocate 40%.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
Using a single GPU on a multi-GPU system:
If you have more than one GPU in your system, the GPU with the lowest ID will be selected by default. 
If you would like to run on a different GPU, you will need to specify the preference explicitly:
'''
# load the label data
data_path = 'C:/behrouz/Research_and_Development/tensorflowLearning/mnist/tfRecordsData/'
test_labels_file = 'test-labels.csv'
train_labels_file = 'train-labels.csv'
# Basic model parameters as external flags.
FLAGS = None
# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'
IMAGE_PIXELS = 784
def _parse_func(example_proto):
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    image =parsed_features['image_raw']
    image = tf.decode_raw(image, tf.uint8)
    image.set_shape([IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image , parsed_features['label']

def inputs(pathToTFRec, data_size, batch_size, isTrain):
    # batching yields a [1, 128] label! reshape and convert to one-hot!
    dataset = tf.data.TFRecordDataset(pathToTFRec, num_parallel_reads=5)# read 5 images at a time from file in Parallel!
    dataset = dataset.map(_parse_func)
    if isTrain:
        dataset = dataset.shuffle(buffer_size=data_size) # shuffle at beginning of epoch!
        dataset = dataset.repeat(count=None)  # repeat indifinitely
        dataset = dataset.batch(batch_size)  # collects a number of examples and stacks them, to create batches.
        dataset = dataset.repeat(count=None)  # repeat indifinitely
    else:
        dataset = dataset.repeat(count=None)  # repeat indifinitely
        dataset = dataset.batch(data_size)
    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    batch_labels = tf.reshape(batch_labels, [-1])
    batch_labels = tf.one_hot(batch_labels, depth=10)
    return batch_images, batch_labels

def model(x, y, reuse=False):
    with tf.variable_scope('Model', reuse=reuse):
        print('Shape of one_hot labels =', y.get_shape())
        layer1 = tf.layers.dense(x, 256, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('layer1', layer1.get_shape())
        layer2 = tf.layers.dense(layer1, 128, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('layer2', layer2.get_shape())

        logits = tf.layers.dense(layer2, 10, activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        # binary true false vector of length of training data
        correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return logits, accuracy


def run_training():
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step(graph=None)
        ########################################################################
        with tf.device('/cpu:0'):
            batch_images, batch_labels = inputs([data_path + TRAIN_FILE],FLAGS.train_size ,128, True)
            validation_images, validaton_labels = inputs([data_path + VALIDATION_FILE], FLAGS.valid_size,
                                                         FLAGS.valid_size, False)
            test_images, test_labels = inputs([data_path + TEST_FILE], FLAGS.test_size, FLAGS.test_size, False)

        print(batch_images.get_shape(), batch_labels.get_shape())
        with tf.device("/device:GPU:0"):
            train_logits, train_accuracy = model(batch_images, batch_labels, reuse=False)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_logits, labels=batch_labels))
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=10000,
                                                       decay_rate=0.5,
                                                       staircase=True,
                                                       name='decaying_learning_rate')
            train_op = tf.contrib.layers.optimize_loss(loss=cost,
                                                       global_step=global_step,
                                                       learning_rate=learning_rate,
                                                       optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                       clip_gradients=20.0,
                                                       name='d_optimize_loss',
                                                       variables=tf.trainable_variables())

            validation_logits, validation_accuracy = model(validation_images, validaton_labels, reuse=True)
            test_logits, test_accuracy = model(test_images, test_labels, reuse=True)



        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # lets check where logits are calculated
            sess.run(cost)
            b,l = sess.run([batch_images, batch_labels])
            print('Shape of Batch=', b.shape, l.shape)
            for i in range(100000):
                start_time = time.time()
                batch_loss, batch_accu, _ = sess.run([cost, train_accuracy, train_op])
                duration = time.time() - start_time
                if i %100==0:
                     print('Step %d: loss = %.2f, accuracy=%.2f (%.3f sec)' % (i, batch_loss,batch_accu, duration))
                     validation_accu = sess.run(validation_accuracy)
                     test_accu = sess.run(test_accuracy)
                     print('Accuracy  Validation= ', validation_accu, 'Test= ', test_accu )
        sess.close()

def main(_):
    run_training()

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default=data_path,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--BATCH_SIZE',
        type=int,
        default=1024,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=50000,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--valid_size',
        type=int,
        default= 5000,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=10000,
        help='Directory with the training data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

