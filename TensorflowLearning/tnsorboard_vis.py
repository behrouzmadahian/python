import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
You can use TensorBoard to visualize your TensorFlow graph, 
plot quantitative metrics about the execution of your graph, 
and show additional data like images that pass through it. 
TensorBoard operates by reading TensorFlow events files, 
which contain summary data that you can generate when running TensorFlow.

1-First, create the TensorFlow graph that you'd like to collect summary data from, 
and decide which nodes you would like to annotate with summary operations.
2- merge the summary functions so that you can run them in graph
3- write these summaries to disk by using tf.summary.FileWriter
The FileWriter takes a logdir in its constructor - this logdir is quite important,
 it's the directory where all of the events will be written out

you could run the merged summary op every single step, and record a ton of training data
consider running the merged summary op every n steps!
launching tensorboard: from command line:
1: tensorboard --logdir=PATH TO LOG DIR
2- python -m tensorflow.tensorboard
'''
#model Global variables defined as flags!!
flags=tf.app.flags
flags.DEFINE_boolean('fake_data',False,'if true, uses fake data for unit testing')
flags.DEFINE_integer('max_steps',500,'Number of training epochs')
flags.DEFINE_float('learning_rate',0.001,'Initial Learning rate')
flags.DEFINE_float('dropout',0.6,'keep probability for training drop out')
flags.DEFINE_string('data_dir','./data/','Directory for storing data')
flags.DEFINE_string('log_dir','./tensorBoard_Vis/','summaries directory for tensorboard code')
FLAGS=flags.FLAGS

def BuildModel_Train():
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)
    sess=tf.Session()
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'): #for visualizing some of the images(10)
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial,tf.float32)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,tf.float32)

    # lets define some summary functions:
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean, variance = tf.nn.moments(var, [0])
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(variance)
                tf.summary.scalar('stddev', stddev)
            with tf.name_scope('minMax'):
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('Histogram', var)

    # defining the model:
    def nn_layer(input_tensor, input_dim, output_dim, layer_name, activation=tf.nn.relu):
        # reusable code for making a simple NN layer!
        # Adding a name scope ensures logical grouping of the layers in the graph!
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('Biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('linear_Output'):
                preactivation = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('Pre-activation-output', preactivation)
            with tf.name_scope('Activations'):
                activs = activation(preactivation)
                tf.summary.histogram('layer-Activations', activs)
        return activs

    hidden1_activations = nn_layer(x, 784, 500, 'layer1', activation=tf.nn.relu)
    hidden2_activations = nn_layer(hidden1_activations, 500, 300, 'layer2',activation=tf.nn.relu)

    with tf.name_scope('Dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('Drop-out-keep-probability', keep_prob)
        dropped = tf.nn.dropout(hidden2_activations, keep_prob)
    # output layer and its output
    y_pred = nn_layer(dropped, 300, 10, 'output_layer', activation=tf.nn.softmax)

    # calculating loss:
    with tf.name_scope('cross-entropy-loss'):
        diff = y_ * tf.log(y_pred)
        cross_entropy = -tf.reduce_mean(diff)
        tf.summary.scalar('corss-entropy-loss', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
    with tf.name_scope('Accuracy'):
        correct_preds = tf.equal(tf.argmax(y_, 1), tf.argmax(y_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

    # merging all summaries:
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)  # save the graph of the model as well!
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')  # save the graph of the model as well!
    init = tf.global_variables_initializer()
    sess.run(init)

    # making a tensorflow feed dict: maps the data into their placeholder:
    def feed_dict(train):
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout

        else:  # test mode:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    # training:
    for i in range(FLAGS.max_steps):
        if i % 10 == 0:
            # record summaries and test accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Test Accuracy at step %s: %s' % (i, acc))
        else:  # record train set summaries and train
            if i % 100 == 99:  # record execution stats:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                      options=run_options, run_metadata=run_metadata)  # train
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary,i)
                print('Adding meta data for ', i)
            else:  # record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main():
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    BuildModel_Train()


if __name__=='__main__':
    main()






