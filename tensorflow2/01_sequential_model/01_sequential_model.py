"""
02-16-20
tf.keras is Tensorflow implementation of the keras API specification. has functionality for:
tf.data pipelines, eager execution, Estimators, ..
It makes tensorflow easier to use without sacrificing performance.

When saving model's weights, tf.keras defaults to the checkpoint format.
Pass: save_format = 'h5' to use HDF5 format, or pass a file name that ends in .h5

Sequential model:
In Keras, you assemble layers to build models. a model is usually a graph of layers
The most common type of model is a stack of layers: the tf.keras.Sequential model

To control dropoout behavior in train and test , use functional API!!!

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import os

# data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0


def create_model():
    model = tf.keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(10, activation='relu'))

    return model


model = create_model()
# logits for first sample:
predictions = model(x_train[:1]).numpy()

# converting logits to probabilities:
probs = tf.nn.softmax(predictions).numpy()
print("logits:\n {}\n probs:\n {}".format(predictions, probs))

# loss: negative log prob of correct class:
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compiling the model before training:
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'], run_eagerly=True)
print('cross entropy loss of first example: {}'.format(loss_fn(y_train[:1], predictions)))

print(model.summary())

# Save model weights to file: during training and at the end of training
checkpoint_folder = '/Users/behrouzmadahian/Desktop/python/tensorflow2/chkpoint_files'
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

checkpoint_path = checkpoint_folder + '/01_keras.ckpt'

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(x_train, y_train, epochs=5, batch_size=128,
          validation_data=(x_test[:1000], y_test[:1000]),
          callbacks=[cp_callback])

eval_results = model.evaluate(x_test, y_test, verbose=2)
print('Test results: {}'.format(eval_results))

# if you want the model to return a probability, you can wrap the trained model and attach the softmax to it
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])


print('\n\n')
print("Predicted probabilities: {}".format(probability_model(x_test[:2])))

# Create a new model with random weights- load model weights and do predictions
model = create_model()
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'], run_eagerly=True)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print('Untrained model accuracy: {:5.2f}%'.format(100*acc))

# load model weights from the checkpoint and re-evaluate
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print('Restored model accuracy: {:5.2f}%'.format(100*acc))

"""
Checkpoint call back options:
The default tensorflow format only saves the 5 most recent checkpoints
save_best_only: save only the best model based on quantity being monitored based
on either the maximization or the minimization of the monitored quantity. for val_acc this should be max
for val_loss this should be min, etc
in auto mode, the direction is automatically inferred from the name of the monitored quantity

save_weights_only:
if False, the full model will be saved
save_best_only=True: Only saves best model  
"""
# save weights every 5 epochs:
# include the epoch in the file name:
checkpoint_path = checkpoint_folder + "/01_keras-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5,
                                                 mode='auto',
                                                 save_best_only=False
                                                 )
tf.keras.backend.clear_session()
# Create new model instance:
model = create_model()
model.compile(optimizer='adam', loss=loss_fn,
              metrics=['accuracy'], run_eagerly=True)
# save model weights prior to training:
model.save_weights(checkpoint_path.format(epoch=0))


model.fit(x_train, y_train, epochs=30, batch_size=256,
          validation_data=(x_test[:1000], y_test[:1000]),
          callbacks=[cp_callback])

# Loading the last check pointed weights:
print('Loading the last save checkpoint file: ')
latest = tf.train.latest_checkpoint(checkpoint_folder)
print('Loaded the last save checkpoint file: ')

print(latest)
model = create_model()
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'],run_eagerly=True)

print('Load previously saved weights to the model..')
print('These weights are loaded using tf.train.latest_checkpoint()')
model.load_weights(latest)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print('Restored model accuracy: {:5.2f}%'.format(100*acc))


"""
Save entire model:
    - model weights     - model configuration(architecure)      - optimizer's configuration

Call model.save to save the model architecure, weights, and training configuration
in a single file/folder
This allows you to export a model so it can be used without access to the original python code.
Since the optimizer state is recovered, you can resume training from exactly where you left off

Saving a fully functional model is very useful- you can load them in Tensorflow.js(HDF5, SavedModel)
and then train and run them in web browsers or convert them to run on mobile devises using Tensorflow Lite

* Custom objects(e.g. subclassed models or layers) require special attention when saving and loading
See the Saving custom objects section below
"""
"""
SavedModel format.
another way to serialize models. 
can be restored using tf.keras.models.load_model
are compatible with tensorflow Serving.
SaverModel guide goes into details about how to serve/inspect the SavedModel
"""
# clear the session:
tf.keras.backend.clear_session()

checkpoint_path = checkpoint_folder + "/01_keras-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 mode='auto',
                                                 save_best_only=True
                                                 )
# HDF5 format
model = create_model()
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'], run_eagerly=True)
model.fit(x_train, y_train, epochs=10, batch_size=256,
          validation_data=(x_test[:1000], y_test[:1000]),
          callbacks=[cp_callback])

model.save(checkpoint_folder+'/01_keras.h5')
model.save(checkpoint_folder+'/01_keras_SavedModelFormat')

# Now recreate the model from that file:

new_model = tf.keras.models.load_model(checkpoint_folder+'/01_keras.h5')
print('Model architecure of model fully loaded from .h5 format: ')
print(new_model.summary())
loss, acc = new_model.evaluate(x_test, y_test)
print('\n\n\n Results of evaluate on the model entirely loaded from .h5 file: ')
print('accuracy: {}'.format(acc))


new_model = tf.keras.models.load_model(checkpoint_folder+'/01_keras_SavedModelFormat')

print('Checking model architecure from model fully loaded from SavedModel format: ')
print(new_model.summary())
loss, acc = new_model.evaluate(x_test, y_test)

print('Accuracy of model loaded from SavedModel format: ')
print(acc)

"""
Saving custom objects
If you are using the SavedModel format, you can skip this section. 
The key difference between HDF5 and SavedModel is that HDF5 
uses object configs to save the model architecture, 
while SavedModel saves the execution graph. 
Thus, SavedModels are able to save custom objects like subclassed models and custom layers 
without requiring the original code.

To save custom objects to HDF5, you must do the following:

1. Define a get_config method in your object, and optionally a from_config class method.
2. get_config(self) returns a JSON-serializable dictionary of parameters needed to recreate the object.
3. from_config(cls, config) uses the returned config from get_config to create a new object. 
4. By default, this function will use the config as initialization kwargs (return cls(**config)).
5. Pass the object to the custom_objects argument when loading the model.
 The argument must be a dictionary mapping the string class name to the Python class. 
 E.g. tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})
"""


