"""
This text classification tutorial trains a recurrent neural network
on the IMDB large movie review dataset for sentiment analysis.
Need to make sure we mask the paddings in Embedding layer so we correctly calculate Y at the end of sequence
here Seq to 1 model
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


# set up input pipeline
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

# This text encoder will reversibly encode any string, falling back to byte-encoding if necessary.
sample_string = 'Hello TensorFlow.'
print('Sample String to encode:')
encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))
assert original_string == sample_string

for index in encoded_string:
    print('{} ----> {}'.format(index, encoder.decode([index])))

BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPOCHS = 1
EMBEDDING_SIZE = 64

padded_shapes = tf.compat.v1.data.get_output_shapes(train_dataset)
print('Padded Shapes for each part of nested dataset : trainx(text), trainy(label)')
print(padded_shapes)

train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes,
                                                                padding_values=None, drop_remainder=False)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)


model = tf.keras.Sequential([
    # Pay attention to mask_zeros argument!
    tf.keras.layers.Embedding(encoder.vocab_size, EMBEDDING_SIZE, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
print(model.summary())

history = model.fit(train_dataset, epochs=EPOCHS,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sample_pred_text, pad=True):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)


# The following predictions MUST give IDENTICAL results:
# Since we correctly Mask ZEROS!
# Predictions without Padding:
sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print("Predictions Without Padding:{}".format(predictions))

# Predictions With padding
predictions = sample_predict(sample_pred_text, pad=True)
print("Predictions With Padding:{}".format(predictions))

# Stack of Two LSTM layers:
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64, mask_zero=True),
    # Pay attention to return_sequences=True
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
print('Stacked LSTM Model:')
print(model.summary())

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=EPOCHS,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)
print('-'*100)
print('Stacked LSTM results: ')
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print("Predictions Without Padding: {}".format(predictions))

# Predictions With padding
predictions = sample_predict(sample_pred_text, pad=True)
print("Predictions With Padding: {}".format(predictions))
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
