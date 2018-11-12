import numpy as np
import tensorflow as tf
import pandas as pd
'''
We will get an overview of how to construct an input function that feeds training, evaluation, and prediction data into
neural networks.
input_fn is used to pass feature and target data to the train, evaluate, and predict methods of the estimator.

anathomy of input_fn:

def my_input_fn():

    # Preprocess your data here...

    # ...then return
     1 ) a mapping of feature columns to Tensors with  the corresponding feature data, and
     2 ) a Tensor containing labels 
       return feature_cols, labels
       
feature_cols:
A dict containing key/value pairs that map feature column names to Tensors
 (or SparseTensors) containing the corresponding feature data.

labels:
A Tensor containing your label (target) values: the values your model aims to predict.

Converting Feature Data to Tensors:
If your feature/label data is a python array or stored in pandas 
dataframes or numpy arrays, you can use the following methods to construct input_fn

# numpy input_fn.
my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_data)},
    y=np.array(y_data),
    ...)

# pandas input_fn.
my_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({"x": x_data}),
    y=pd.Series(y_data),
    ...)
    
Passing input_fn Data to Your Model
classifier.train(input_fn=my_input_fn, steps=2000)

Note that the input_fn parameter must receive a function object 

However, it is very advantagous to be able to pass parameters so that 
we have one input function for train evaluation and test!

Wrap the input function in a function that accepts the parameters and returns the input function!
#EXAMPLE 1:
def get_input_fn_from_pandas(dataset_X,dataset_y, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({"x": data_set_X}),
      y=pd.Series(dataset_y)
      num_epochs=num_epochs,
      shuffle=shuffle)

#EXAMPLE 2:
def get_input_fn_from_numpy(data_set_x,dataset_y, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.numpy_input_fn(
      x={'x':np.array(data_set_x)},
      y=np.array(dataset_y),
      num_epochs=num_epochs,
      shuffle=shuffle)

Here we build a NN to predict the House values in Boston!

'''
#define the column names of the data set in csv file.
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",  "dis", "tax", "ptratio", "medv"]
features=COLUMNS[:-1]
label=COLUMNS[-1]


#defining input_fn:
def get_input_fn(dataset,num_epochs=None,shuffle=True):
    dataset=dataset.values
    dataset_x=dataset[:,:-1]; dataset_y=dataset[:,-1]
    input_fn=tf.estimator.inputs.numpy_input_fn(
            x={'x': np.array(dataset_x)},
            y=np.array(dataset_y),
        num_epochs=num_epochs,
        shuffle=shuffle
    )
    return input_fn

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.zeros(shape=shape)
    return tf.Variable(initial)

def model_fn(features,labels,mode,params):
    #two layer MLP of size defined in params!
    # lw1=weight_variable(params['size'][:2])
    # lb1=bias_variable([params['size'][1]])
    #
    # l1out_preac=tf.matmul(tf.cast(features['x'], tf.float32),lw1)+lb1
    #
    # l1out=tf.nn.tanh(l1out_preac)
    # l1_loss=tf.nn.l2_loss(lw1)+tf.nn.l2_loss(lb1)
    #
    # lw2=weight_variable(params['size'][1:])
    # lb2=bias_variable(params['size'][1])
    # l2out_preac=tf.matmul(l1out,lw2)+lb2
    # l2out=tf.nn.tanh(l2out_preac)
    # l2_loss=tf.nn.l2_loss(lw2)+tf.nn.l2_loss(lb2)
    #
    # Wout=weight_variable([params['size'][1],1])
    # bout=bias_variable(1)
    #
    # output=tf.matmul(l2out,Wout)+bout
    # predictions = tf.reshape(output, [-1])


    layer1=tf.layers.dense(features['x'],params['size'][0],activation=tf.nn.tanh,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2Reg']))

    layer2=tf.layers.dense(layer1,params['size'][1],activation=tf.nn.tanh,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2Reg']))

    output=tf.layers.dense(layer2,1) #no activation!

    predictions=tf.reshape(output,[-1],name='predss') #flattening the predictions

    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions={'House-med-value':predictions})

    #defining loss:

    myloss=tf.losses.mean_squared_error(labels,predictions)

    #defining training op:
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    training_op=optimizer.minimize(loss=myloss,global_step=tf.train.get_global_step())

    if mode==tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,loss=myloss,train_op=training_op)

    #defining metrics and put them in eval_metric dict!
    #Note that the labels tensor is cast to a float64 type to match the data type of the predictions tensor
    #which will contain real values

    eval_metric_ops={
        'rmse':tf.metrics.root_mean_squared_error(
            tf.cast(labels,tf.float64),predictions)
    }

    return tf.estimator.EstimatorSpec(mode=mode,loss=myloss,eval_metric_ops=eval_metric_ops)

params={'learning_rate':0.001,'size':(10,10),'l2Reg':10.}

def main(unused_argv):
    # reading data:
    training_set = pd.read_csv('boston_train.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv('boston_test.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv('boston_predict.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
    # defining feature columns for the input data which formally specify the set of features to use for training
    # Because all features in the housing data set contain continuous values, you can create their FeatureColumns
    # using the tf.contrib.layers.real_valued_column()

    #I dont know how to explicityly use feature_cols in my defined model_fun put in estimator!
    feature_cols = [tf.feature_column.numeric_column(k) for k in features]

    # creating some logging hooks:
    tensors_to_log = {'Preds': 'predss'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)

    myModel=tf.estimator.Estimator(model_fn=model_fn,model_dir='./median-housingPrediction',params=params)
    myModel.train(input_fn=get_input_fn(training_set,num_epochs=None,shuffle=True),steps=2000, hooks=[logging_hook])

    #evaluating the model:
    eval_results=myModel.evaluate(input_fn=get_input_fn(test_set,num_epochs=1,shuffle=False))
    print(eval_results)

    #prediction:
    predictions=myModel.predict(input_fn=get_input_fn(prediction_set,num_epochs=1,shuffle=False))
    print(predictions)




if __name__=='__main__':
    tf.app.run()


