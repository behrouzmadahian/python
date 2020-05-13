"""
we can access the activations of intermediate layers ("nodes" in the graph) and reuse them elsewhere.
This is extremely useful for feature extraction, for example.
This comes in handy when implementing neural style transfer, among other things.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import VGG19
vgg19 = VGG19()

# intermediate activations of the model obtained by querying the graph structure
features_list = [layer.output for layer in vgg19.layers]
# we can use these features to create a new feature extraction model, that returns the values of
# intermediate layer's activations
feature_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
print(feature_extraction_model.summary())
img = np.random.random((1, 224, 224, 3)).astype('float32')
extracted_features = feature_extraction_model(img)
print('Shape of extracted features: ')
for feat in extracted_features:
    print(feat.shape)

