"""
options:
- We can save the model + optimizer state (SavedModel or h5 format)
- We can only save model weights .ckpt

Metrics and saving model:
- Before saving the model, reset the model metrics- this way model before and after loading will have
the same state(starting at ZERO??,)
- - Metric states are not preserved by Model.Save_weights
"""
import numpy as np
from tensorflow import keras

inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_MLP_model')
print(model.summary())

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.
x_test = x_test.reshape(10000, 784).astype('float32') / 255.

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.RMSprop()

model.compile(loss=loss_fn, optimizer=optimizer)
history = model.fit(x_train, y_train, batch_size=64, epochs=1)

# Save model predictions for future checks:
predictions = model.predict(x_test)

"""
Saving whole model:
- You can save a model built with the functional API into a single file. 
- you can later recreate the same model from this file, even if you no longer have access to the
code that created the model. This file includes:

- The Model's architecture
- The model's weight values
- The model's training config (what you passed to compile) if any
- The optimizer and its state if any (this enables you to restart training from where you left off)
"""
# Saving to .h5 format:
checkpoint_folder = 'PATHTO CHECKPOINT FOLDER'
model_name = 'my_model.h5'
model.save(checkpoint_folder+model_name)

# re-create the exact same model purely from the file
new_model = keras.models.load_model(checkpoint_folder + model_name)

# Check that state is preserved:
new_predictions = new_model.predict(x_test)
all_close = np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
print('New Predictions from loaded model from .h5 are identical to predictions before model.save: ({})'.format(all_close))
# Note that the optimizer state is preserved as well. you can resume training where you left off

"""
Export to SavedModel format
You can also export a whole model to tensorflow SavedModel format.
SavedModel is a standalone serialization format for Tensorflow objects
supported by tensorflow serving as well as tensorflow implementations other than Python

The SavedModel files that were created contain:
- A Tensorflow checkpoint containing the model weights
- A SavedModel proto containing the underlying tensorflow graph
"""
# Export the model to SavedModel
moel_namec = 'my_model'
model.save(checkpoint_folder + model_name, save_format='tf')
# re-create the exact same model:
new_model = keras.models.load_model(checkpoint_folder + model_name)

# Check that the state is preserved:
new_predictions = model.predict(x_test)
all_close = np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
print('New Predictions loaded model from SavedModel are identical to predictions before model.save: ({})'.format(all_close))

