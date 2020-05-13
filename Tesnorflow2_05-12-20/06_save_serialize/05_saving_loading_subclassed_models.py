"""
Sequential models and Functional models are data structures that represent a DAG of layers.
As such, they can be safely serialized and deserialized.

A subclassed model differs in that it's not a data structure, it's a piece of code.
The architecture of the model is defined via the body of the call method. 
This means that the architecture of the model cannot be safely serialized.
To load a model, you'll need to have access to the code that created it (the code of the model subclass). 
Alternatively, you could be serializing this code as bytecode (e.g. via pickling),
but that's unsafe and generally not portable.

First of all, a subclassed model that has never been used cannot be saved.
That's because a subclassed model needs to be called on some data in order to create its weights.

Until the model has been called, it does not know the shape and dtype of the input data it should be expecting,
and thus cannot create its weight variables. 
- You may remember that in the Functional model from the first section, 
  the shape and dtype of the inputs was specified in advance (via keras.Input(...)) 
  -- that's why Functional models have a state as soon as they're instantiated.
  
There are three different approaches to save and restore a subclassed model.
The following sections provides more details on those three approaches.
"""
import numpy as np
from tensorflow import keras
import tensorflow as tf


class ThreeLayerMLP(keras.Model):
  def __init__(self, name=None):
    super(ThreeLayerMLP, self).__init__(name=name)
    self.dense_1 = keras.layers.Dense(64, activation='relu', name='dense_1')
    self.dense_2 = keras.layers.Dense(64, activation='relu', name='dense_2')
    self.pred_layer = keras.layers.Dense(10, name='predictions')
  
  def call(self, inputs):
    x = self.dense_1(inputs)
    x = self.dense_2(x)
    return self.pred_layer(x)


def get_model():
  return ThreeLayerMLP(name='3_layer_mlp')


model = get_model()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.
x_test = x_test.reshape(10000, 784).astype('float32') / 255.

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.RMSprop()
model.compile(loss=loss_fn, optimizer=optimizer)
history = model.fit(x_train, y_train, batch_size=64, epochs=1)
"""
Reset Metrics before saving so that loaded model has the same state
since metrics states are not preserved by Model.save_weights()
"""
model.reset_metrics()
checkpoint_folder = 'PATH to Checkpoint Folder'
model_name = 'my_model'
##############
# Approach 1 #
##############
"""
The recommended way to save a subclassed model is to use save_weights() to create a tensorflow SavedModel checkpoint
WHich contain the value of all variables associated with the model:
- the layers' weights
- The optimizer's state
- Any variable associated with stateful model metrics(if any)

To restore your model, you will need access to the code that created the model object.
Note that in order to restore the optimizer state and the state of any stateful metric, 
you should compile the model (with the exact same arguments as before) and call it on some data before calling load_weights
"""
model.save_weights(checkpoint_folder + model_name, save_format='tf')
predictions = model.predict(x_test)
# save the loss on the first batch to later
# assert that optimizer state was preserved
first_batch_loss = model.train_on_batch(x_train[:64], y_train[:64])

new_model = get_model()
new_model.compile(loss=loss_fn, optimizer=optimizer)

# Call the new model on some data -> it will initialize the variables as well as stateful metrics and optimizer
new_model.train_on_batch(x_train[:1], y_train[:1])

# load the state of the old model:
new_model.load_weights(checkpoint_folder+model_name)

# check model state is preserved:
new_predictions = new_model.predict(x_test)
# raises assetion error if False!
try:
    np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
    print('New Predictions from loaded model from .h5 are identical to predictions before model.save: ({})')

except:
  print('New Predictions from loaded model from .h5 are NOT identical to predictions before model.save: ({})')

# The optimizer state is preserved as well
# So you can resume training where you left off
new_first_batch_loss = new_model.train_on_batch(x_train[:64], y_train[:64])
assert first_batch_loss == new_first_batch_loss

##############
# Approach 2 #
##############
"""
Use model.save to save whole model and by using load_model to restore 
previously stored subclassed model.
"""
model.save(checkpoint_folder+'myModel')
# re-create exact same model purely from file:
new_model = keras.models.load_model(checkpoint_folder+'myModel')

##############
# Approach 3 #
##############
"""
Use tf.saved_model.save()
This is equivalent to the tf format in model.save.
"""
tf.saved_model.save(model, checkpoint_folder+'myModel')
# restore the model:
restored_saved_model = keras.models.load_model(checkpoint_folder+'myModel')

