import tensorflow as tf
import random, os, time
import argparse, sys
'''
We converted our data into TFREcords for train, validation, and test data.
Now we want to design the pipeline to read these data!!
'''
# load the label data
data_path = 'C:/behrouz/Research_and_Development/tensorflowLearning/mnist/tfRecordsData'
test_labels_file = 'test-labels.csv'
train_labels_file = 'train-labels.csv'
# Basic model parameters as external flags.
FLAGS = None
# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'
IMAGE_PIXELS = 784

def read_and_decode(filename_queue):
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    '''
    As part of the API, a TFRecordReader always acts on a queue of filenames. It will pop a filename off the queue
    and use that filename until the tfrecord is empty. At this point it will grab the next filename off the filename queue.

    '''
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(serialized_example,
                                     # Defaults are not specified since both keys are required.
                                     features={
                                         'image_raw': tf.FixedLenFeature([], tf.string),
                                         'label': tf.FixedLenFeature([], tf.int64),
                                              }
                                     )
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
      # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    return image, label

def inputs(mode, batch_size, num_epochs, shuffle):
  """Reads input data num_epochs times.
  Args:
    mode: Selects between the training, validation, or test data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  if mode == 'train':
      filename = os.path.join(FLAGS.train_dir, TRAIN_FILE)
  elif mode =='validation':
      filename = os.path.join(FLAGS.train_dir, VALIDATION_FILE)
  else:
      filename = os.path.join(FLAGS.train_dir, TEST_FILE)

  with tf.name_scope('input'):
      # shuffle makes sense if we hame multiple tfRecord files!
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs, shuffle=shuffle)

    # Even when reading in multiple threads, share the filename queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if mode =='train':
        images, sparse_labels = tf.train.shuffle_batch([image, label],
                                          batch_size=batch_size,
                                          num_threads=2,
                                          capacity=1000 + 3 * batch_size,
                                          # Ensures a minimum amount of shuffling of examples.
                                          min_after_dequeue=1000)
    elif mode =='validation':
        images, sparse_labels = tf.train.batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=2,
                                                       capacity=batch_size)
    else:
        images, sparse_labels = tf.train.batch([image, label],
                                               batch_size=batch_size,
                                               num_threads=2,
                                               capacity=batch_size)
    sparse_labels =tf.one_hot(sparse_labels, depth=10)
    return images, sparse_labels

#
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

        train_images, train_labels = inputs('train', FLAGS.BATCH_SIZE,10, True)
        print(train_labels.get_shape(),'===')
        validation_images, validaton_labels = inputs('validation', FLAGS.valid_size, 10, False)
        test_images, test_labels = inputs('test', FLAGS.test_size, 10, False)
        train_logits, train_accuracy = model(train_images, train_labels, reuse=False)
        validation_logits, validation_accuracy =model(validation_images, validaton_labels, reuse=True)
        test_logits, test_accuracy = model(test_images, test_labels, reuse=True)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_logits, labels=train_labels))
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=10000,
                                                   decay_rate=0.5,
                                                   staircase=True,
                                                   name='decaying_learning_rate')
        optimizer = tf.contrib.layers.optimize_loss(loss=cost,
                                                  global_step=global_step,
                                                  learning_rate= learning_rate,
                                                  optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                  clip_gradients=20.0,
                                                  name='d_optimize_loss',
                                                  variables=tf.trainable_variables())
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord =coord)
            try:
                step = 0
                while not coord.should_stop():
                    start_time = time.time()
                    batch_loss, batch_accu, _ =  sess.run([cost, train_accuracy, optimizer])
                    duration = time.time()-start_time
                    if step %100==0:
                        print('Step %d: loss = %.2f, accuracy=%.2f (%.3f sec)' % (step, batch_loss,batch_accu, duration))
                        validation_accu = sess.run(validation_accuracy)
                        test_accu = sess.run(test_accuracy)
                        print('Accuracy  Validation= ', validation_accu, 'Test= ', test_accu )
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

                # Wait for threads to finish.
            coord.join(threads)
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
        default=128,
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
