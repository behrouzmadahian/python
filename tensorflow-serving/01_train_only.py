import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
Trains the model, Saved the EMA weights to file.
The check point will be loaded later to export the servable.
This can be used for online/batch training, and keep in mind to control the version.
'''
tf.app.flags.DEFINE_integer('training_iteration', 1000, 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version',1, 'version number of the model.')
# ubuntu on windows: Note how the path is defined!
tf.app.flags.DEFINE_string('checkpoint_dir',
                           '/mnt/c/behrouz/PythonCodes/ML/tensorflow-serving/models/mnist/mnist-train/',
                           'Working directory.')
FLAGS = tf.app.flags.FLAGS
if not os.path.exists(FLAGS.work_dir + '/' + str(FLAGS.model_version)):
    os.makedirs(FLAGS.work_dir + '/' + str(FLAGS.model_version))


def model(x):
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # sess.run(tf.global_variables_initializer())
    logits = tf.nn.softmax(tf.matmul(x, w) + b, name='y_score')
    return logits


def main(_):
    # sys.argv: gives the command line arguments provided when running from commandline.
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: MNIST-train-export-toServe.py [--training_iteration=x] '
              '[--model_version=y] export_dir')
        sys.exit(-1)
    # Train model
    print('Training model...')
    # data can come from online source, local disk, .. parent of workdir
    datadl_to = os.path.abspath(os.path.join(FLAGS.work_dir, os.pardir))
    mnist = input_data.read_data_sets(datadl_to, one_hot=True)
    # defining input signature:
    # Return a tensor with the same shape and contents as input<tf_example['x']>.
    # defining input signature:
    '''
    # serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    # feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32)}
    # tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    '''
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    logits = model(x)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(logits), axis=1))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    # EMA weights:
    with tf.control_dependencies([train_step]):
        train_op_new = ema.apply(tf.trainable_variables())
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(FLAGS.training_iteration):
            batch = mnist.train.next_batch(50)
            sess.run(train_op_new, feed_dict={x: batch[0], y_: batch[1]})
        print('training accuracy %g' % sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
        print('Done training!')
        saver.save(sess, FLAGS.work_dir + str(FLAGS.model_version) + '/train.ckpt')


if __name__ == '__main__':
    tf.app.run()