"""
You can write you own training and evaluation loops: Using GradientTape
Calling a model inside a GradientTape scope enables you to retrieve the gradients of 
the trainable weights of the layer with respect to a loss value.
we can use these gradients to update these variables which we can retrieve using model.trainable_weights
"""
from tensorflow import keras
import tensorflow as tf


def create_model():
    inputs = keras.Input(shape=(784,), name='digits')
    x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = keras.layers.Dense(10, name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = create_model()
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
batch_size = 64
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Training loop:
epochs = 3
for epoch in range(epochs):
    print('Start of epoch: {}'.format(epoch))
    # Iterate over the batches of the data set
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Open a GradientTape to record the operations run
        # During forward pass, which enables auto differentiation
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer
            # The operations that the layer applies to its input are going to be recorded on the GradientTape
            logits = model(x_batch_train, training=True)  # run the model in training mode <drop out enabled!
            loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 200 == 0:
                print('Training loss (for one batch) at step {}: {}'.format(step, float(loss_value)))
                print('Seen so far {} . Samples'.format((step + 1) * 64))
