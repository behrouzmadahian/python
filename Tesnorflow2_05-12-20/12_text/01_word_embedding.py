"""
This tutorial introduces word embeddings. It contains complete code to train word embeddings
from scratch on a small dataset, and to visualize these embeddings using the Embedding Projector
Unfortunately, this does not do unsupervised learning of word representations and
learns embeddings to the task which is not as interesting.

The Embedding layer:
 Can be understood as a lookup table that maps from integer indices
 (which stand for specific words) to dense vectors (their embeddings).

mask_zero: Whether or not the input value 0 is a special "padding" value that should be masked out.
This is useful when using recurrent layers which may take variable length input.
If this is True then all subsequent layers in the model need to support masking
or an exception will be raised. If mask_zero is set to True, as a consequence,
index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).

input_length: Length of input sequences, when it is constant. This argument is required
if you are going to connect Flatten then Dense layers upstream
(without it, the shape of the dense outputs cannot be computed).

padding_values: (Optional.) A nested structure of scalar-shaped tf.Tensor, representing the padding values to
 use for the respective components. Defaults are 0 for numeric types and the empty string for string types.
Input shape:
For text or sequence problems:
the Embedding layer takes a 2D tensor of integers, of shape (samples, sequence_length),
where each entry is a sequence of integers. It can embed sequences of variable lengths.
e.g. You could feed into the embedding layer above batches with shapes (32, 10)
(batch of 32 sequences of length 10)


The returned tensor has one more axis than the input,
the embedding vectors are aligned along the new last axis. Pass it a (2, 3) input batch and the output is (2, 3, N)
(samples, sequence_length, embedding_dimensionality).
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# vocab size 1000, embedding vector len = 5:
# If mask zero is true, need to make sure the first word in vocab is not encoded as ZERO!
embedding_layer = keras.layers.Embedding(1000, 5, mask_zero=True)
# embedding vectors of words with indices 1, 2 , 3
result = embedding_layer(tf.constant([1, 2, 3]))
print('Output and Shape of embedding  on one sample of length 3:')
print(result.numpy(), result.shape)

result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
print('Output and Shape of embedding  on one batch of shape of length 3:')

print(result.shape)

(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',
                                          split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                          with_info=True, as_supervised=True)
print(info)

# since data is Nested, trainx, trainy, we need to tell padded_batch how each of these must be padded
output_shapes_train = tf.compat.v1.data.get_output_shapes(train_data)
padded_shapes = output_shapes_train  # (TensorShape([None]), TensorShape([]))
print('Padded Shapes for each part of nested dataset : trainx(text), trainy(label)')
print(padded_shapes)
"""
Get the encoder (tfds.features.text.SubwordTextEncoder), and have a quick look at the vocabulary.

The "_" in the vocabulary represent spaces. Note how the vocabulary includes whole words (ending with "_") 
and partial words which it can use to build larger words:

in padded_batch:
drop_remainder:
A tf.bool scalar tf.Tensor, representing whether the last batch should be dropped in the case it has fewer
 than batch_size elements; the default behavior is not to drop the smaller batch.
"""
encoder = info.features['text'].encoder
print(encoder.subwords[:20])

# Movie reviews can be different lengths. We will use the padded_batch method to standardize the lengths of the reviews.
train_batches = train_data.shuffle(1000).padded_batch(batch_size=10, padded_shapes=padded_shapes,
                                                      padding_values=None, drop_remainder=False)
test_batches = test_data.shuffle(1000).padded_batch(batch_size=10, padded_shapes=padded_shapes)

"""
As imported, the text of reviews is integer-encoded 
(each integer represents a specific word or word-part in the vocabulary).
"""
print('batch Shape and labels:')
train_batch, train_labels = next(iter(train_batches))
print(train_batch.numpy().shape)
print('---')
print(train_labels.numpy())

"""
Create a simple model
We will use the Keras Sequential API to define our model. In this case it is a "Continuous bag of words" style model.

Next, the Embedding layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index.
These vectors are learned as the model trains. The vectors add a dimension to the output array.
The resulting dimensions are: (batch, sequence, embedding_len).

Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example
by averaging over the sequence dimension. This allows the model to handle input of variable length, 
in the simplest way possible.

This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.

The last layer is densely connected with a single output node. Using the sigmoid activation function, 
this value is a float between 0 and 1, representing a probability (or confidence level) that the review is positive.
With this approach our model reaches a validation accuracy of around 88% 
(note the model is over fitting, training accuracy is significantly higher).
"""
embedding_dim = 16

model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, embedding_dim),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(16, activation='relu'),
  keras.layers.Dense(1)
])

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_batches, epochs=10,
                    validation_data=test_batches, validation_steps=20)

import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5, 1))
plt.show()

# Retrieve the learned embeddings:
# This will be a matrix of shape (vocab_size, embedding-dimension).
e = model.layers[0]
weights = e.get_weights()[0]
print('Shape of Embedding Matrix:')
print(weights.shape)

"""
We will now write the weights to disk. To use the Embedding Projector, 
we will upload two files in tab separated format: a file of vectors 
(containing the embedding), and a file of meta data (containing the words).
Then Load these two files into Embedding projector:
http://projector.tensorflow.org/
This can be run on local tensorboard instance as well
I am unable to Visualize these embeddings into Embedding Projector!!!
files simply do no load up!

"""
import io

encoder = info.features['text'].encoder

out_v = io.open('/Users/behrouzmadahian/Desktop/python/tensorflow2/12_text/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('/Users/behrouzmadahian/Desktop/python/tensorflow2/12_text/meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
    vec = weights[num+1]  # skip 0, it's padding.
    if num == 0:
        print(vec.shape)
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
