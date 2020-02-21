"""
In general, you will use the layer class to define inner computation blocks, 
and you will use the Model class to define the outer model-- the object you will train
For instance in the ResNet50 model, you would have several ResNet blocks subclassing Layer,
and a single Model encompassing the entire ResNet50 network

The model class has the same API as Layer, with the following differences:
- it exposes built-in training, evaluation, and prediction loops
    model.fit(), model.evaluate(), and model.predict()
- it exposes the list of its inner layers, via model.layers property
- it exposes saving and serialization APIs

Effectively the 'Layer' class corresponds to what we refer to in the literature as a "layer" or as a 'blocl' 
(as in inception block or resnet block)
Meanwhile, the "Model" class corresponds to what is referred to in the literature as a "model" or "network
In other words, a Model is just like a Layer with added train, evaluate, test, and serialization utilities
"""
