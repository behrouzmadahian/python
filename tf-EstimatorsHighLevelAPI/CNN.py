import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
'''
'layers' module provides a high level API that makes it easy to construct a NN.
it provides modules that facilitate the creation of Dense, and conv layers, adding activation functions and
apply dropout regularization.
The model we build here has 2 convolutional layes and two dense layer!
We create the model, in a function that conforms to tensorflow's Estimator API.

Input Layer:
Conv2d and pooling layers expect input tensors of shape:
[batch_size,image_width, image_height, channels]
if instead of batch_size, we put -1, it means this dimension should be dynamically computed(can be set later!)

#Dropout Layer:
dropout=tf.layers.dropout(inputs=dense,rate=0.4,training=mode==tf.estimator.ModeKeys.TRAIN):

rate: percentage of units to randomly drop during training

training argument: 
    takes a boolean specifying whether or not the model is currently being run in training mode,
    -dropout will only be applied if training is true!
    here we check if the mode passed to our cnn_model_fn is TRAIN model

#Generate Predictions:
creates predicted class and the probabilities! for each possible target class

Class predictions and probabilities are put in a  dictionary and   returned as and EstimatorSpec() object

#Configuring the training operation: look at the code!

'''

def cnn_model_fn(features,labels,mode,params):

    #mode is to see if it is training mode or test mode!
    #Input layer:
    #data is flattened so we need to convert it back to image dimensions
    #we will create a tensorflow input function in main!
    input_layer=tf.reshape(features['x'],[-1,28,28,1])

    #conv layer 1:
    conv1=tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5,5],strides=1,
                           padding='same',activation=tf.nn.relu)

    #pooling layer 1:
    pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

    #conv layer 2:
    conv2=tf.layers.conv2d(inputs=pool1, filters=64,kernel_size=[5,5],strides=1,
                           padding='same',activation=tf.nn.relu)

    #pooling layer 2
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    #dense layer:
    pool2_flat=tf.reshape(pool2,[-1,7*7*64]) #we need to calculate what are the shapes at pool 2 to use here!
    dense=tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)

    #I will explain on this more below!

    dropout=tf.layers.dropout(inputs=dense,rate=0.4,training=mode==tf.estimator.ModeKeys.TRAIN)

    #logit layer:  for efficiency we apply softmax activation later!!!
    logits=tf.layers.dense(inputs=dropout, units=10)

    predictions={
        #generate predictions for PREDICT and EVAL mode
        'classes':tf.argmax(input=logits, axis=1),

        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
        'probabilities': tf.nn.softmax(logits,name='softmax_tensor')
    }

    #in prediction mode, only classes and probability of each class is returned!
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    #calculate Loss for both train and eval mode
    #Because labels contain a series of values from 0–9, indices is
    #  just our labels tensor, with values cast to integers.
    onehot_labels=tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)

    loss=tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)

    #Configure the Training op (for TRAIN mode):

    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op=optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    #Add evaluation metrics(for EVAL mode):
    eval_metric_ops={
        'accuracy': tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])
    }
    #if mode is not train or predict, the following will be returned!
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

learning_rate=0.001

def main(unused_argv):
    #load training data:
    mnist=tf.contrib.learn.datasets.load_dataset('mnist')
    train_data=mnist.train.images #numpy array
    train_labels=np.asarray(mnist.train.labels,dtype=np.int32)

    test_data=mnist.test.images #numpy array
    test_labels=np.asarray(mnist.test.labels,dtype=np.int32)

    #create the Estimator:
    #create an Estimator (a TensorFlow class for performing high-level model
    # training, evaluation, and inference) for our model.
    #The model_fn argument specifies the model function to use for training, evaluation, and prediction

    # Set model params
    model_params = {"learning_rate": learning_rate}
    #we also specify the dircectory where model data(checkpoints) will be saved

    mnist_classifier=tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='./mnist_conv_model',params=model_params)

    #set up a Logging hook:
    #Since CNNs can take a while to train, let's set up some logging so we can track progress during training.

    #We can use TensorFlow's tf.train.SessionRunHook to create a tf.train.LoggingTensorHook that will
    #  log the probability values from the softmax layer of our CNN.
    #We store a dict of the tensors we want to log in tensors_to_log.
    #  Each key is a label of our choice that will be printed in the log output,
    #  and the corresponding label is the name of a Tensor in the TensorFlow graph.

    tensors_to_log={'probabilities':'softmax_tensor'}
    logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)

    #Training the model:
    #creating train_input_fn and calling train() on mnist_classifier
    #pass training data as a dict (only x and not responses)!
    #num_epochs=None: the model is run untill a specified number of steps (training on batches is reached!
    #shuffle=True: shuffles the training data

    train_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    mnist_classifier.train(input_fn=train_input_fn, steps=2000, hooks=[logging_hook])

    #Evaluating the model:
    #Once training is complete, we want to evaluate our model to determine its accuracy on the test set.
    # we call evaluate method which evaluates the metrics we specified in 'eval_metric_ops'
    #of the cnn_model_fn

    eval_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False )

    eval_results=mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    
if __name__=='__main__':
    tf.app.run()



