import tensorflow as tf
import time
import argparse
import sys
'''
We converted our data into TFRecords for train, validation, and test data.
Now we want to design the pipeline to read these data!!
If we create a Dataset using a placeholder for data, we can dynamically change the  data
inside the Data set -> this is helpful when we have Validation and Train and we want to switch between
the two and both are say in TFRecords format and we create data set from each!
************Improvements made here:*************
    Explanations:
    adding more parallelization to the input pipeline
    a. Parallelize Data Extraction: <tf.contrib.data.parallel_interleave>
    In a real-world setting, the input data may be stored remotely, and it would not make sense to replicate the
    input data on every machine. A dataset pipeline that works well when reading data locally might become
    bottlenecked on I/O when reading data remotely because of the following differences between
    local and remote storage:
    Time-to-first-byte: Reading the first byte of a file from remote storage can take orders of magnitude
    longer than from local storage.
    Read throughput: While remote storage typically offers large aggregate bandwidth, reading a single file might
    only be able to utilize a small fraction of this bandwidth.
    
    To mitigate the impact of the various data extraction overheads, the
    tf.data API offers the <tf.contrib.data.parallel_interleave> transformation 
    to mix together records from N different shards
    Use this transformation to parallelize the execution of and interleave the contents of other datasets
    (such as data file readers). 
    The number of datasets to overlap can be specified by the <cycle_length> argument.
    If your pre-processing increases the size of your data, we recommend applying the interleave,
    prefetch, and shuffle first (if possible) to reduce memory usage.
    
    b. We recommend applying the shuffle transformation before the repeat transformation,
    ideally using the <fused shuffle_and_repeat> transformation.
    
    NOT DONE HERE:
    c. If your data can fit into memory, use the cache transformation to cache it in memory during the first epoch,
     so that subsequent epochs can avoid the overhead associated with reading, parsing, and transforming it.
    https://www.tensorflow.org/performance/datasets_performance
    Improvements made here:
     1. read files in parallel and interleave the examples read <parallelize extraction>.
     2. use the fused operation:  tf.contrib.data.shuffle_and_repeat
     use tf.contrib.data.map_and_batch() to:
     3. Parallelize Data Transformation
     4. parallelize batch creation
     use dataset.prefetch(batch_size) to: 
     5. # prefetch next batch while training on curr batch
     Note: batching yields labels of shape [1, batch_size], you need to flatten and turn to one hot if necessary!
'''
# load the label data
data_path = 'C:/behrouz/PythonCodes/ML/TensorflowLearning/mnist/tfRecordsData/'
# Basic model parameters as external flags.
FLAGS = None
# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train-*.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'
IMAGE_PIXELS = 784


def _parse_func(example_proto):
    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    image = parsed_features['image_raw']
    image = tf.decode_raw(image, tf.uint8)
    image.set_shape([IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, parsed_features['label']


def inputs(filenames, batch_size, is_train):
    # batching yields a [1, 128] label! reshape and convert to one-hot!
    # num_parrallel_reads: number of tfRecord files to read in parallel
    # filenames: The filenames argument to the TFRecordDataset initializer. can
    # either be a string, a list of strings, or a tf.Tensor of strings.
    # 1. read files in parallel and interleave the examples read.
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=FLAGS.num_parallel_readers)
    if is_train:
        # shuffle the items as they pass through when size become: data_size(each epoch)
        # it maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer.
        dataset = dataset.shuffle(buffer_size=10000)
    # 2. parallelize data transformation
    dataset = dataset.map(_parse_func, num_parallel_calls=FLAGS.num_parallel_calls)
    # The repeat method restarts the Dataset when it reaches the end- end of epoch.
    dataset = dataset.repeat(count=None)
    # collects a number of examples and stacks them, to create batches.
    dataset = dataset.batch(batch_size)
    # prefetch data while training on current batch- perf optimization
    # the prefetch transformation will yield benefits any time there is an
    # opportunity to overlap the work of a "producer" with the work of a "consumer."
    dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)
    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    batch_labels = tf.reshape(batch_labels, [-1])
    batch_labels = tf.one_hot(batch_labels, depth=10)
    return batch_images, batch_labels

# this function has all the optimization guidelines except caching!


def inputs_train():
    files = tf.data.Dataset.list_files(data_path+'train-*.tfrecords', shuffle=True)
    # 1. read files in parallel and interleave the examples read.
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset,
        cycle_length=FLAGS.num_parallel_readers))
    # shuffle the items as they pass through when size become: data_size(each epoch)
    # it maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer.
    # count= None: repeat forever; 10000: min number of elements in the buffer before shuffling
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000, count=None))
    # 2. Parallelize Data Transformation
    # 3. parallelize batch creation
    # if your batch size is in the hundreds or thousands, your pipeline will likely
    #  additionally benefit from parallelizing the batch creation.
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_parse_func,
        batch_size=FLAGS.batch_size,
        num_parallel_batches=None,  # can not set both num_..
        num_parallel_calls=20  # if None:= batch_size * num_parallel_batches
        ))
    # prefetch next batch while training on curr batch
    dataset = dataset.prefetch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    batch_labels = tf.reshape(batch_labels, [-1])
    batch_labels = tf.one_hot(batch_labels, depth=10)
    return batch_images, batch_labels


def model(x, y, reuse=False):
    with tf.variable_scope('Model', reuse=reuse):
        print('Shape of one_hot labels =', y.get_shape())
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


def run_training():
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step(graph=None)
        with tf.device('/cpu:0'):
            batch_images, batch_labels = inputs_train()
            validation_images, validaton_labels = inputs([data_path + VALIDATION_FILE],
                                                         FLAGS.valid_size,
                                                         False)
            test_images, test_labels = inputs([data_path + TEST_FILE],
                                              FLAGS.test_size,
                                              False)
        print('Shape of batch= ')
        print(batch_images.get_shape(), batch_labels.get_shape())
        with tf.device('/device:GPU:0'):
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
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            b, l = sess.run([batch_images, batch_labels])
            print('Shape of Batch=', b.shape, l.shape)
            for i in range(100000):
                start_time = time.time()
                batch_loss, batch_accu, _ = sess.run([cost, train_accuracy, train_op])
                duration = time.time() - start_time
                if i % 100 == 0:
                    print('Step %d: BATCH loss = %.2f, accuracy=%.2f (%.3f sec)' % (i, batch_loss, batch_accu, duration))
                    validation_accu = sess.run(validation_accuracy)
                    test_accu = sess.run(test_accuracy)
                    print('Accuracy  Validation= ', validation_accu, 'Test= ', test_accu)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
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
        '--batch_size',
        type=int,
        default=256,
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
        default=5000,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--prefetch_buffer_size',
        type=int,
        default=1024,
        help='# of samples to Pipeline while training on current batch.'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=10000,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--num_parallel_calls',
        type=int,
        default=20,  # number of CPU cores
        help='number of examples to transform in parallel using tf.data.Dataset.map'
    )
    parser.add_argument(
        '--num_parallel_readers',
        type=int,
        default=10,  # number of CPU cores
        help='number of files to read in parallel and iterleave the examples from those.'
    )
    parser.add_argument(
        '--num_parallel_batches',
        type=int,
        default=10,  # number of CPU cores
        help='number of batches to create in parallel.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)