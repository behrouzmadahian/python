from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import matplotlib.pyplot as plt

"""
download and prepare Cifar10:
60k color images in 10 classes, 6k per class.
50k train and 10k test
"""
(train_img, train_labels), (test_img, test_labels) = keras.datasets.cifar10.load_data()
# normalize pixel values between 0, 1:
train_img, test_img = train_img / 255.0, test_img / 255.0

# plotting 25 images from this dataset:
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i], cmap=plt.cm.binary)
    # CFAR labels happens to be arrays
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
# plt.show()


def get_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model()
print(model.summary())
optimizer = keras.optimizers.Adam(1e-3)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
history = model.fit(train_img, train_labels, epochs=10, validation_data=(test_img, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_img, test_labels, verbose=2)
print("Accuracy on test data {:7.2f}".format(test_acc))
