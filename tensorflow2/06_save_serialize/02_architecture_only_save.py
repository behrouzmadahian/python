"""
Sometime you are interested in architecutre of the model and you don't need to save the weights or optimizer.
- In this case you can retrieve the "config" of the model via get_config() method
- The config is a Python dict that enables you to re-create the same model -- initialized from scratch,
without any of the information learned previously during training
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

# Save model predictions for future checks:
predictions = model.predict(x_test)

# Getting model config:
config =  model.get_config()
print('Model CONFIG: \n {}'.format(config))
reinitialized_model = keras.Model.from_config(config)
new_predictions = reinitialized_model.predict(x_tesall_close = np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
print('New Predictions Reinitialized model Identical to presave? ({})'.format(all_close)))

"""
You can alternatively use  to_json() and from_json(), which uses a JSON string to store the
config instead of a python dict.
This is useful to save the config to disk
"""
json_config = model.to_json()
reinitialized_model = keras.models.model_from_json(json_config)


