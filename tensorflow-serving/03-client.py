import sys
import threading
import grpc
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/mnt/c/behrouz/PythonCodes/ML/tensorflow-serving/models/mnist/',
                           'Working directory.')
tf.app.flags.DEFINE_string('result_dir', '/mnt/c/behrouz/PythonCodes/ML/tensorflow-serving/models/mnist/mnist-results/',
                           'result directory')
FLAGS = tf.app.flags.FLAGS

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

'''
Example IO:
define the image url to be sent to the model for prediction
image_url = "https://www.publicdomainpictures.net/pictures/60000/nahled/bird-1382034603Euc.jpg"
# or get it from local disk!
response = requests.get(image_url)
image = np.array(Image.open(StringIO(response.content)))
height = image.shape[0]
width = image.shape[1]
'''


def do_inference(hostport, work_dir, num_tests):
    """Tests PredictionService with concurrent requests.
    Args:
      hostport: Host:port address of the PredictionService.
      work_dir: The full path of working directory for test data set.
      num_tests: Number of test images to use.
    Returns:
      The classification error rate.
      writes the predictions to file.
    Raises:
      IOError: An error occurred processing test data set.
    """
    # need to get the data from a url or local drive, ..
    test_data_set = input_data.read_data_sets(work_dir).test
    testx, testy = test_data_set.images[:num_tests], test_data_set.labels[:num_tests]

    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    pred_list = np.zeros((num_tests, 2))
    for i in range(num_tests):
        request = predict_pb2.PredictRequest()
        # if you desire results of specific model version:
        # request.model_spec.version.value = 2
        # since we define model name here, in calling tensorflow_model_server we need to have option:
        #model_name=mnist
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'predict_images'
        image, label = np.reshape(testx[i], (1, len(testx[i]))), testy[i]
        # images and logits names defined in 01_export_Serve.py
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
        result_future = stub.Predict.future(request, 5.0)  # time out=5 seconds
        response = np.array(
            result_future.result().outputs['logits'].float_val)
        response = np.argmax(response)
        pred_list[i] = label, response
    accuracy = np.mean(np.equal(pred_list[:, 0], pred_list[:, 1]).astype(float))
    np.savetxt(FLAGS.result_dir + 'test_results.csv', pred_list, delimiter=',')
    return accuracy


def main(_):
    if FLAGS.num_tests > 10000:
        print('num_tests should not be greater than 10k')
        return
    if not FLAGS.server:
        print('please specify server host:port')
        return
    accuracy = do_inference(FLAGS.server,
                            FLAGS.work_dir,
                            FLAGS.num_tests)
    print('\nInference Accuracy: %s%%' % (accuracy * 100))


if __name__ == '__main__':
    tf.app.run()

