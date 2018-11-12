import argparse, sys,tempfile
#import urllib
from six.moves import urllib
FLAGS=None
import numpy as np
import tensorflow as tf
#enable logging:
tf.logging.set_verbosity(tf.logging.INFO)



'''
tf.estimator framework makes it easy o construct and train ML models via its high level ESTIMATOR API
ESTIMATOR offers classes you can instantiate to quickly configure common model types
such as regressors and classifiers:

    tf.estimator.LinearClassifier: Constructs a linear classification model.
    tf.estimator.LinearRegressor: Constructs a linear regression model.
    tf.estimator.DNNClassifier: Construct a neural network classification model.
    tf.estimator.DNNRegressor: Construct a neural network regression model.
    tf.estimator.DNNLinearCombinedClassifier: Construct a neural network and linear combined classification model.
    tf.estimator.DNNRegressor: Construct a neural network and linear combined regression model.

But what if none of tf.estimator's predifined model types meet your needs?
e.g the ability to use costomized loss , or other model structures,...
Here we cover how to create our OWN Estimator using building blocks provided in tf.estimator
We will learn the following:

    Instantiate an Estimator
    Construct a custom model function
    Configure a neural network using tf.feature_column and tf.layers
    Choose an appropriate loss function from tf.losses
    Define a training op for your model
    Generate and return predictions
    
note: LSTM layer does not exist in tf.layers. Only dense and convolution layers.
However we can mix and combine these as we wish (I think!- will get back to this when I cover RNNs)


Problem to solve: estimate age of abalone(sea snail) by the number of rings on its shell.
#data has 7 features
'''
#Downloading csv files and loading into tensorflow Dataset
#define  a func that loads the csv files:
def maybe_download(train_data, test_data, predict_data):
    if train_data:
        train_file_name=train_data
    else:
        #Return a file-like object that can be used as a temporary storage area.
        #tempfile.NamedTemporaryFile: generates temporary files and directories!
        #the file is guaranteed to have a visible name
        #That name can be retrieved from the -name- attribute of the returned file-like object
        train_file=tempfile.NamedTemporaryFile(delete=False)

        urllib.request.urlretrieve("http://download.tensorflow.org/data/abalone_train.csv",train_file.name)

        train_file_name=train_file.name
        train_file.close()
        print('Training data is downloaded to %s'%train_file_name)
    if test_data:
        test_file_name=test_data
    else:
        test_file=tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("http://download.tensorflow.org/data/abalone_test.csv",test_file.name)
        test_file_name=test_file.name
        test_file.close()
        print("Test data is downloaded to %s" % test_file_name)

    if predict_data:
        predict_file_name=predict_data
    else:
        predict_file=tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("http://download.tensorflow.org/data/abalone_predict.csv",predict_file.name)
        predict_file_name=predict_file.name
        predict_file.close()
        print("Prediction data is downloaded to %s" % predict_file_name)
    return train_file_name,test_file_name,predict_file_name


#constructing our  estimator:
#when you're creating your own estimator from scratch, the constructor accepts
# just two high-level parameters for model configuration, < model_fn and params >

#model function must contain all the logic to support training, evaluation and prediction.
#we need to implement all these functionality

#params: an optional dictionary of parameters such as learning rate, dropout,... that will be passed to the model_fn
#
#Just like tf.estimator's predefined regressors and classifiers, the Estimator
# initializer also accepts the general configuration arguments     < model_dir and config >
'''
def model_fn(features, labels, mode, params):

   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
   return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)

features: a dictionary containing the features passed to the model via input_fn
labels: contains labes passed to the model via input_fn < will be empty for predict() calls -> Model will infer this!

mode: one of the following tf.estimator.ModeKeys string values indicating the context in which the model_fn
was invoked:
    tf.estimator.ModeKeys.TRAIN The model_fn was invoked in training mode, namely via a train() call.
    tf.estimator.ModeKeys.EVAL. The model_fn was invoked in evaluation mode, namely via an evaluate() call.
    tf.estimator.ModeKeys.PREDICT. The model_fn was invoked in predict mode, namely via a predict() call.

model_fn may also accept a params argument containing a dict of hyperparameters used for training 

The model_fn must return a tf.estimator.EstimatorSpec object, which contains the following values:

1-mode (required). The mode in which the model was run. Typically, you will return the mode argument of the model_fn here.

2-predictions (required in PREDICT mode). A dict that maps key names of your choice to 
Tensors containing the predictions from the model, e.g.:

predictions = {"results": tensor_of_predictions}

In PREDICT mode, the dict that you return in EstimatorSpec will then be returned by predict(),
 so you can construct it in the format in which you'd like to consume it.
 
3-loss (required in EVAL and TRAIN mode). A Tensor containing a scalar loss value: 
the output of the model's loss function (discussed in more depth later in Defining loss for the model) 
calculated over all the input examples. This is used in TRAIN mode for error handling and logging, 
and is automatically included as a metric in EVAL mode.

4-train_op (required only in TRAIN mode). An Op that runs one step of training.

5-eval_metric_ops (optional). A dict of name/value pairs specifying the metrics that will be calculated 
when the model runs in EVAL mode. The name is a label of your choice for the metric,
 and the value is the result of your metric calculation. 
 The tf.metrics module provides predefined functions for a variety of common metrics. 
 The following eval_metric_ops contains an "accuracy" metric calculated using tf.metrics.accuracy:

eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels, predictions) }

If you do not specify eval_metric_ops, only loss will be calculated during evaluation.


Input Layer data:
The input layer is a series of nodes (one for each feature in the model) that will accept the feature data that
 is passed to the model_fn in the features argument. 
 If features contains an n-dimensional Tensor with all your
    feature data, then it can serve as the input layer.
 If features contains a dict of feature columns passed to the model via an input function,
    you can convert it to an input-layer Tensor with the tf.feature_column.input_layer function.
    e.g:
    input_layer = tf.feature_column.input_layer(
    features=features, feature_columns=[age, height, weight])

Here, because you'll be passing the abalone Datasets using numpy_input_fn as shown below, 
features is a dict {"x": data_tensor}, so features["x"] is the input layer. 

Supplementary metrics for evaluation can be added to an eval_metric_ops dict. 
    
'''
learning_rate=0.001

def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])

  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec( mode=mode,  predictions={"ages": predictions})
  #Defining Loss:
  #There are several predefined losses and you can define your own loss using tensors available.
  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(labels, predictions)

  #you can define metrics and put them into eval_metric_ops dict!

  #Note that the labels tensor is cast to a float64 type to match the data type of the predictions tensor
  #which will contain real values

  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float64), predictions)
  }
  #defining training op for the model
  #when training, typically, the goal is to minimize loss!.
  #For global_step, the convenience function tf.train.get_global_step takes care of generating an integer variable!

  optimizer=tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

  train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

  # Provide an estimatorSpec for 'ModeKeys.EVAL' and 'ModeKeys.TRAIN' modes.

  return tf.estimator.EstimatorSpec(
                            mode=mode,
                            loss=loss,
                            train_op=train_op,
                             eval_metric_ops=eval_metric_ops)


#creating main() and loading CSV data into datasets:
#additionally, we define flags so that we can provide the data sets from command line and run the code!

def main(unused_argv):
    #load datasets:
    #note that since FLAGS.train_data,.. are empty, it tells maybe_donwload function to download them into a temp file!
    abalone_train, abalone_test, abalone_predict = maybe_download(
        FLAGS.train_data, FLAGS.test_data, FLAGS.predict_data)
    #training Examples:
    training_set=tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_train, target_dtype=np.int, features_dtype=np.float64 )

    # Test examples
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

    # Set of 7 examples for which to predict abalone ages
    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

    # Set model params
    model_params = {"learning_rate": learning_rate}

    # Instantiate Estimator
    myEstimator=  tf.estimator.Estimator(model_fn=model_fn, params=model_params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train
    myEstimator.train(input_fn=train_input_fn, steps=5000)

    # Score accuracy
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    ev = myEstimator.evaluate(input_fn=test_input_fn)
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])

    # Print out predictions
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": prediction_set.data},
        num_epochs=1,
        shuffle=False)
    predictions = myEstimator.predict(input_fn=predict_input_fn)
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i + 1, p["ages"]))


if __name__ == "__main__":

    #The argparse module makes it easy to write user-friendly command-line interfaces.
    #The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv

  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_data", type=str, default="", help="Path to the training data.")
  parser.add_argument(
      "--test_data", type=str, default="", help="Path to the test data.")
  parser.add_argument(
      "--predict_data",
      type=str,
      default="",
      help="Path to the prediction data.")
  FLAGS, unparsed = parser.parse_known_args()
  print('*'*10)
  print(sys.argv[0])
  print(unparsed)
  print(FLAGS)
  print(FLAGS.train_data,'===')
  print('*'*10)

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


