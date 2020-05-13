"""
TensorFlow Hub is a way to share pre-trained model components.
See the TensorFlow Module Hub for a searchable listing of pre-trained models.

This tutorial demonstrates:
- How to use TensorFlow Hub with tf.keras.
- How to do image classification using TensorFlow Hub.
- How to do simple transfer learning.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import PIL.Image as Image
from matplotlib import pyplot as plt
import numpy as np

# Download the classifier
"""
Use hub.module to load a mobilenet, and tf.keras.layers.Lambda to wrap it up as a keras layer.
Any TensorFlow 2 compatible image classifier URL from tfhub.dev will work here.
"""
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_SHAPE = (224, 224)
print(IMAGE_SHAPE + (3,))
classifier = keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))])

print(classifier.summary())
# Download a single image and run the model on it!
img_address = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'

grace_hopper = tf.keras.utils.get_file('image.jpg', img_address)
# resizing the image to match input of the mobilenet!
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper)/255.0

print("Shape of the image: after resize: {}".format(grace_hopper.shape))

pred_results = classifier.predict(grace_hopper[np.newaxis, ...])
# results are logits across 1001 classes!
print("Shape of prediction vector: {}".format(pred_results.shape))
pred_class = np.argmax(pred_results[0], axis=-1)
print('predicted class: {}'.format(pred_class))
# decode the predictions:
# We have the predicted class ID, Fetch the ImageNet labels, and decode the predictions
labels_path = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', labels_path)
imagenet_labels = np.array(open(labels_path).read().splitlines())
print(imagenet_labels)

plt.imshow(grace_hopper)
plt.axis('off')
pred_class_name = imagenet_labels[pred_class]
print(pred_class_name)
plt.title('Prediction: ' + pred_class_name.title())  # Capitalize the first char in word
plt.show()

"""
Using TF Hub it is simple to retrain the top layer of the model to recognize the classes in our dataset
"""
new_data_path = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_root = tf.keras.utils.get_file('flower_photos', new_data_path, untar=True)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
image_data = image_generator.flow_from_directory(str(data_root), batch_size=32, target_size=IMAGE_SHAPE)

# the resulting object is an iterator that returns image_batch, labels_batch pairs
for image_batch, label_batch in image_data:
    print('Image batch shape: ', image_batch.shape)
    print('Label batch shape: ', label_batch.shape)
    break

# run the classifier on a batch of images:
result_batch = classifier.predict(image_batch)
print('result_batch shape: {}'.format(result_batch.shape))
predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
print('predicted classes:\n {}'.format(predicted_class_names))

plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6, 5, n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
  plt.suptitle("ImageNet predictions")
plt.show()
"""
The results are far from perfect, but reasonable considering that 
these are not the classes the model was trained for (except "daisy").
"""
"""
Download the headless model
TensorFlow Hub also distributes models without the top classification layer. 
These can be used to easily do transfer learning.

Any Tensorflow 2 compatible image feature vector URL from tfhub.dev will work here.
"""
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3))
feature_batch = feature_extractor_layer(image_batch)
print("Shape of features out of headless mobilenet: {}".format(feature_batch.shape))

# Freeze the variables in the feature extractor layer, so that the training only modifies the new classifier layer
feature_extractor_layer.trainable = False
# attach a classifier head
model = keras.Sequential([feature_extractor_layer, keras.layers.Dense(image_data.num_classes)])
print(model.summary())
predictions = model(image_batch)
print('Shape of predictions -logits- before fine tuning final layer :{}'.format(predictions.shape))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# To visualize the training progress, use a custom callback to log the loss and
# accuracy of each batch individually, instead of the epoch average.
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['accuracy'])
    self.model.reset_metrics()  # this will reset the metrics so we cannot get the epoch end loss and metrics!!


steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
batch_stats_callback = CollectBatchStats()
model.fit_generator(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[batch_stats_callback])

print(batch_stats_callback.batch_losses)
print(batch_stats_callback.batch_acc)
fig, ax = plt.subplots(2, 1, sharex='col')
ax[0].plot(batch_stats_callback.batch_losses)
ax[0].set_title('batch Losses during training')
ax[1].plot(batch_stats_callback.batch_acc)
ax[1].set_title('batch Accuracy during training')

plt.show()
# To redo the plot from before, first get the ordered list of class names:
print(image_data.class_indices.items())
class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
class_names = np.array([key.title() for key, value in class_names])
print(class_names)

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6, 5, n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
  plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.show()
# Export your model:
import time
t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
model.save(export_path, save_format='tf')

print('Model saved to: {}'.format(export_path))

# reload the model:
reloaded = keras.models.load_model(export_path)
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)
print('Are there any difference in prediction of fine tuned model before and after reloading from file: ')
print(abs(reloaded_result_batch - result_batch).max())
# This saved model can be loaded for inference later, or converted to TFLite or TFjs.

