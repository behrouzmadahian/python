"""
Sometime you are interested in Model weights and not in architecture.
In this case you can retrieve the weights values as a list of numpy arrays via get_weights() 
and set the model weights set_weights

You can combine get_config();  from_config() and get_weights, set_weights() to re-create your model in the same state
However, unlike model.save(), this will not include the training config and the optimizer.
- you would have to call compile again before using the model for training

-- the save-to-disk alternative to get_weights() and set_weights() is
save_weights(PATH) and load_weights(PATH)
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

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
predictions = model.predict(x_test)

# Save JSON config to disk:
checkpoint_folder = 'PATH to Checkpoint Folder'
config_name = 'model_config.json'
# check to see if you can load the model wiehgts from .ckpt
checkpoint_file = 'model.h5'
json_config = model.to_json()
with open(checkpoint_folder+config_name, 'w') as json_file:
    json_file.write(json_config)

# Save weights to disk
model.save_weights(checkpoint_folder+checkpoint_file)

# Re-load the model from 2 files we saved:
with open(checkpoint_folder+config_name) as json_file:
    json_config = json_file.read()

new_model = keras.models.model_from_json(json_config)
new_model.load_weights(checkpoint_folder+checkpoint_file)
new_predictions = new_model.predict(x_test)
all_close = np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
print('New Predictions from loaded model from .h5 are identical to predictions before model.save: ({})'.format(all_close))

##############
# check to see if you can load the model weights from .ckpt
checkpoint_file = 'model.ckpt'
json_config = model.to_json()
with open(checkpoint_folder+config_name, 'w') as json_file:
    json_file.write(json_config)

# Save weights to disk
model.save_weights(checkpoint_folder+checkpoint_file)

# Re-load the model from 2 files we saved:
with open(checkpoint_folder+config_name) as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights(checkpoint_folder+checkpoint_file)

new_predictions = new_model.predict(x_test)
all_close = np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
print('New Predictions from loaded model from .h5 are identical to predictions before model.save: ({})'.format(all_close))
