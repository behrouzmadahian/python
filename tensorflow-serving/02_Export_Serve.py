'''
Model is trained using 01_train_only.py.
01_train_only can be used to initially train and then online and/or batch train
I separated the train, to be able to load EMA weights.

Loads the weights from a checkpoint file(EMA weights).
Builds the predict, .. signatures and saves the servable to file!
We could use this for training as well, However, I chose to use the 01_train_only.py for this purpose.

Export a simple Softmax Regression TensorFlow model.
Uses TensorFlow SavedModel to export the trained model
with proper signatures that can be loaded by standard
tensorflow_model_server.
Usage: mnist_saved_model.py  [--model_version=y] export_dir

Input Output (IO) must be designed to meet the needs of project and IS project specific.
'''
import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
# ubuntu on windows: Note how the path is defined!
tf.app.flags.DEFINE_string('work_dir', '/mnt/c/behrouz/PythonCodes/ML/tensorflow-serving/models/mnist/mnist-serve/',
                           'Working directory.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/mnt/c/behrouz/PythonCodes/ML/tensorflow-serving/models/mnist/mnist-train/',
                           'checkpoint directory')
FLAGS = tf.app.flags.FLAGS

# make sure the version does not exits if it does set the version to a number bigger that the max directory name
# directory names are integers
versions = sorted(np.array([o for o in os.listdir(FLAGS.work_dir)
                            if os.path.isdir(os.path.join(FLAGS.work_dir, o))]).astype(int))


def model(x):
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    logits = tf.nn.softmax(tf.matmul(x, w) + b, name='y_score')
    return logits


def main(_):
    # sys.argv: gives the command line arguments provided when running from commandline.
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: MNIST-train-export-toServe.py [--model_version=y] export_dir')
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)

    if len(versions) >= 1:
        if FLAGS.model_version <= versions[-1]:
            print('please provide model_version which is equal to "max model_version on file + 1"')
            sys.exit(-1)
    # defining input signature:
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32)}
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    # use tf.identity() to assign name
    # Return a tensor with the same shape and contents as input<tf_example['x']>.
    x = tf.identity(tf_example['x'], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    logits = model(x)
    # finds values and indices of the top k largest entries
    # why not just using argmax and getting the predicted class?
    values, indices = tf.nn.top_k(logits, 10)
    # this set up returns the prediction classes as index of type string.
    table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(10)]))
    prediction_classes = table.lookup(tf.to_int64(indices))
    # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Getting EMA var names:
    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    # EMA weights:
    ema_dict = {}
    for var in tf.trainable_variables():
        ema_var_name = ema.average_name(var)
        ema_dict[ema_var_name] = var
    saver = tf.train.Saver(ema_dict)

    with tf.Session() as sess:
        # need this for initializing shaddow variables for train_step_new!
        sess.run(tf.global_variables_initializer())
        # restore from previous model!
        saver.restore(sess, FLAGS.checkpoint_dir + str(FLAGS.model_version) + '/train.ckpt')

        # Export model
        # the export base path provided in commandline:
        export_path_base = sys.argv[-1]
        export_path = os.path.join(tf.compat.as_bytes(export_path_base),
                                   tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        # Builds the SavedModel protocol buffer and saves variables and assets.
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # Build the signature_def_map.
        classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
        classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
        classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs
                },
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: classification_outputs_scores
                },
                method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

        tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(logits)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'logits': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        # Adds the current meta graph to the SavedModel and saves variables.
        # This function assumes that the variables to be saved have been initialized.
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images': prediction_signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
            },
            legacy_init_op=legacy_init_op,
            # Op or group of ops to execute when the graph is loaded. Note that
            # when the main_op is specified it is run after the restore op at load-time.
            main_op=None,
            # An instance of tf.train.Saver that will be used to export the metagraph.
            #  If None, a sharded Saver that restores all variables will be used.
            saver=None
        )
        builder.save()
        print('Done exporting!')


'''
After training run the server with the following command > go to th directory first!
python mnist_client.py --num_tests=1000 --server=localhost:9000
'''
if __name__ == '__main__':
    tf.app.run()