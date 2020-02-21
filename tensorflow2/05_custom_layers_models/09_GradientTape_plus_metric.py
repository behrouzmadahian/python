"""
You can write you own training and evaluation loops: Using GradientTape
Calling a model inside a GradientTape scope enables you to retrieve the gradients of 
the trainable weights og the layer with respect to a loss value.
we can use these gradients to update these variables which we can retrieve using model.trainable_weights
Lets add metrics to the mix.
You can readily reuse the builtin metrics(or custim ones yoou wrote) in such training loops written from scratch:
1. Instantiate the metric at the start of the loop
2. Call metric.update_state() after each batch
3. Call metric.result() when you need to display the current value of the metric
4. Call metric.reset_states() when you need to clear the state of the metric (at the end of the epoch)
"""
from tensorflow import keras
import tensorflow as tf


def create_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = keras.layers.Dense(64, activation'relu', name='dense_1')(inputs)
  x = keras.layers.Dense(64, activation'relu', name='dense_2')(x)
  outputs = keras.layers.Dense(10, name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model


model = create_model()
optimizer = keras.optimizer.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
batch_size = 64

# Prepare the metrics:
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Loading data
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

# Training loop:
epochs = 3
for epoch in range(epochs):
  print('Start of epoch: {}'.format(epoch))
  # Iterate oveer the batches of the data set
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    # Open a GradientTape to record the operations run
    # During forward pass, which enables auto differentiation
    With tf.GradientTape() as tape:
      # Run the forward pass of the layer
      # the operations that the layer applies to its input are going to be recorded on the GradientTape
      logits = model(x_batch_train, training=True) # run the model in training mode <drop out enabled!
      loss_value = losss_fn(y_batch_train, logits)
      grads = tape.gradient(loss_value, model.trainable_weights)
      # Run one step of gradient descent by updating the value of the variables to minimize the loss
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
      
      # Update training Metric:
      train_acc_metric(y_batch_train, logits)
      
      if step %200 == 0:
        print('Training loss (for one batch) at step {}: {}'.format(step, float(loss_value)))
        print('Seen so far {} . Samples'.format((step + 1 ) * 64))
      
  # Display metrics at the end of each epoch
  train_acc = train_acc_metric.result()
  print('Training acc over epoch: {}'.format(float(train_acc)))
  # Reset training metrics at the end of each epoch:
  train_acc_metric.reset_states()

  # Run a validation loop at the end of each epoch:
  for x_batch_val, y_batch_val in val_dataset:
    val_logits = model(x_batch_val, training=None)
    # Update val metrics:
    val_acc_metric(y_batch_val, val_logits)
  val_acc = val_acc_metric.result()
  val_acc_metric.reset_states()
  print('Validation Accuracy: {}'.format(float(val_acc)))
