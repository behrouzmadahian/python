"""
When writing the call method of a layer, you can create loss tensors 
that you will want to use later, when writing your training loop
This is doable by calling self.add_loss(value).
These losses(including those created by any inner layer) can be retrieved via:
layer.losses
This property is RESET at the start of every __call__ to the top-level layer so that
layer.losses ALWAYS contains the loss values created during the last forward pass

"""
import tensorflow as tf
from tensorflow import keras


class ActivityRegularizerLayer(keras.layers.Layer):
  def __init__(self, rate=1e-4):
    super(ActivityRegularizerLayer, self).__init__()
    self.rate=rate
  
  def call(self, inputs):
    self.add_loss(self.rate * tf.reduce_sum(inputs))
    return inputs
  
  
class OuterLayer(layes.Layer):
  def __init__(self):
    super(OuterLayer, self).__init__()
    self.activity_reg = ActivityRegularizerLayer(1e-3)
  
  def call(self, inputs):
    return self.activity_reg(inputs)

  
layer = OuterLayer()
assert len(layer.losses) == 0 # no losses yet since the layer has never been called
_  = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1 # We created one loss value

# layer.losses gets reset at the start of each __call__
_ = layer(tf.ones(1, 1))
assert len(layer.losses) == 1 # this is the loss created during the call above

# These losses are meant to be taken into account when writing training loops like this:
def create_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = keras.layers.Dense(64, activation'relu', name='dense_1')(inputs)
  # Insert activity regularization layer to regularize the first layer!
  x = ActivityRegularizationLayer()(x)
  x = keras.layers.Dense(64, activation'relu', name='dense_2')(x)
  outputs = keras.layers.Dense(10, name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare train data set:
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation data set:
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
optimizer = keras.optimizers.SGD(learning_rate=1e-4)
loss_fn = keras.losses.SparseCategoricalCrossEntropy(from_logits=True)
model = create_model()
model.compile(optimizer=optimizer, loss=loss_fn)

for x_batch_train, y_batch_train in train_dataset:
  with tf.GradientTape() as tape:
    logits = model(x_batch_train, training=True)
    loss_value = loss_fn(y_batch_train, logits)
    # add Extra losses created during this forward pass:
    loss_value += sum(model.losses)
  grads = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  
  
