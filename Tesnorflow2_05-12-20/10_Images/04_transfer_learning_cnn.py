"""
classify images of cats and dogs using transfer learning from pre-trained network
Use TensorFlow Datasets to load the cats and dogs dataset.

This tfds package is the easiest way to load pre-defined data. If you have your own data,
and are interested in importing using it with TensorFlow see loading image data.
The tfds.load method downloads and caches the data, and returns a tf.data.Dataset object.
These objects provide powerful, efficient methods for manipulating data and piping it into your model.

MobileNet V2:
pretrained covnet trained on ImageNet dataset- 1.4M images and 1000 classes!

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
                                                'cats_vs_dogs',
                                                split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                with_info=True,
                                                as_supervised=True,
                                            )
print(raw_train)
print(raw_validation)
print(raw_test)
# Show the first two images and labels from the training set
get_label_name = metadata.features['label'].int2str
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
plt.show()

# format the Data:
# use the tf.image module to format the images for the task
IMG_SIZE = 160  # all images will be resized to 160 * 160


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# Apply this function to each item in the dataset using the map method:
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
# Now shuffle and batch the data:
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
for image_batch, label_batch in train_batches.take(1):
    print('Shape of image batch and label batch- TRAIN data:')
    print(image_batch.shape)
    print(label_batch.shape)
    break

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
# include_top = False: removes the final classification layer!
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
feature_batch = base_model(image_batch)
print(feature_batch.shape)
# set the weights of the base model to be non trainable -  freeze them!
base_model.trainable = False
print(base_model.summary())
# all a classification head

global_average_layer = keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print("Shape of prediction batch: {}".format(prediction_batch.shape))
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

base_learning_rate = 0.0001
model.compile(optimizer=keras.optimizers.RMSprop(base_learning_rate),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())
print('Number of trainable variable objects: {}'.format(len(model.trainable_variables)))

# pre finetuning performance:
initial_epochs = 10
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
print('Initial Loss: {:.2f}'.format(loss0))
print('Initial accuracy: {:.2f}'.format(accuracy0))

history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

"""
One way to increase performance even further is to train (or "fine-tune") the weights of the top 
layers of the pre-trained model alongside the training of the classifier you added. The training 
process will force the weights to be tuned from generic feature maps to features associated 
specifically with the dataset.
Note: 
This should only be attempted after you have trained the top-level classifier with the 
pre-trained model set to non-trainable. If you add a randomly initialized classifier on top of
a pre-trained model and attempt to train all layers jointly, the magnitude of the gradient 
updates will be too large (due to the random weights from the classifier) and your pre-trained 
model will forget what it has learned.
Also, you should try to fine-tune a small number of top layers rather than the whole MobileNet model. 

In most convolutional networks, the higher up a layer is, the more specialized it is. The first few 
layers learn very simple and generic features that generalize to almost all types of images. 
As you go higher up, the features are increasingly more specific to the dataset on which the 
model was trained. The goal of fine-tuning is to adapt these specialized features to work with 
the new dataset, rather than overwrite the generic learning.

# Unfreeze the top layers of the model:

All you need to do is unfreeze the base_model and set the bottom layers to be un-trainable. 
Then, 
********you should recompile the model (necessary for these changes to take effect), and resume training.

This technique is usually recommended when the training dataset is large and very similar to the original 
dataset that the pre-trained model was trained on.
"""
base_model.trainable = True
print('Number of layers in the base model: ', len(base_model.layers))
fine_tune_at = 100  # fine tune layers from this layer onward!
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


# Continue training the model
# if you trained to convergence earlier, this step will improve the accuracy by a few percentage points!
print('Epoch at which to resume training: {}'.format(history.epoch[-1]))
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])
print('Len of trainable variables(Kernels + biases! for finetuning: {}'.format(len(model.trainable_variables)))
# Compile the model using a much lower learning rate.
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(train_batches, epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_batches)
"""
Let's take a look at the learning curves of the training and validation accuracy/loss
when fine-tuning the last few layers of the MobileNet V2 base model and training 
the classifier on top of it. 
The validation loss is much higher than the training loss, so you may get some overfitting.

You may also get some overfitting as the new training set is relatively small and similar 
to the original MobileNet V2 datasets.

After fine tuning the model nearly reaches 98% accuracy.
"""
# add to the previous list of metrics and losses the visualize!
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.axvline(x=initial_epochs-1, color='green')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.axvline(x=initial_epochs-1, color='green')

plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
