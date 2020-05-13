"""
Check to see how come we just pass the value of the loss to the Mean metric!!!
Mean metric: averages batch losses!!
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras 

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)
print(x_test.shape)
batch_size = 32
buffer_size = 10000
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


class Mymodel(keras.Model):
  def __init__(self):
    super(Mymodel, self).__init__()
    self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
    self.flatten = keras.layers.Flatten()
    self.d1 = keras.layers.Dense(128, activation='relu')
    self.d2 = keras.layers.Dense(10)
  
  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return x


model = Mymodel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and accuracy of the model. 
# These metrics accumulate the values over epochs and then print the overall results

train_loss_metric = keras.metrics.Mean(name='train_loss_metric')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss_metric = keras.metrics.Mean(name='test_loss_metric')
test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout, BN).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # update the metrics accumulators-states:
    train_loss_metric(loss)
    train_accuracy(labels, predictions)
    

@tf.function
def test_step(images, labels):
  """
  training=False is only needed if there are layers with different
  behavior during training versus inference (e.g. Dropout).
  """
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)
  print('Shape of test loss coming out of loss object: {}'.format(t_loss))
  test_loss_metric(t_loss)
  test_accuracy(labels, predictions)
  
  
EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss_metric.reset_states()
  train_accuracy.reset_states()
  test_loss_metric.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss_metric.result(),
                        train_accuracy.result()*100,
                        test_loss_metric.result(),
                        test_accuracy.result()*100))
  
