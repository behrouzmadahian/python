"""

Stacked Recurrent layers in Decoder and Encoder:
Need to investigate how to properly perform:
1. How the attention is built?
    What layers of encoder and decoder are involved
2. How to feed the encoder state to decoder?
    From what layer to what layer

3. How to build the Context vector to feed into decoder alongside input!!!

sequence to sequence (seq2seq) model for Spanish to English translation.
We will look into the generated attention plot.
This shows which parts of the input sentence has the model's attention while translating
takes 10 min to run on P100 GPU

We'll use a language dataset provided by http://www.manythings.org/anki/.
This dataset contains language translation pairs in the format:

May I borrow this book? ¿Puedo tomar prestado este libro?

After downloading the dataset, here are the steps we'll take to prepare the data:

1. Add a start and end token to each sentence.
2. Clean the sentences by removing special characters.
3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
4. Pad each sentence to a maximum length.
"""
from __future__ import print_function, absolute_import, division, unicode_literals
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time

url = 'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip'
# Download the file
path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin=url, extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"
print('The file will be downloaded to: \n{}'.format(path_to_file))


# convert unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii((w.lower().strip()))
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print('Original English sentence: {}'.format(en_sentence))
print("Preprocessed Eng sentence: {}".format(preprocess_sentence(en_sentence)))
print('Original Spanish sentence: {}'.format(sp_sentence))
print("Preprocessed Spanish sentence: {}".format(preprocess_sentence(sp_sentence).encode('utf-8')))
print('-'*50)


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return sentence pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  # divide the Eng and Spanish and preprocess!
  sent_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

  sent_pairs = zip(*sent_pairs)
  return sent_pairs


en, sp = create_dataset(path_to_file, None)
print('English and Spanish pairs after creating dataset: ')
print(en[:4])
print(sp[:4])


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# Limit the size of the dataset to experiment faster (optional)
# Try experimenting with the size of that dataset
num_examples = 30000
input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset(path_to_file, num_examples)

print('Data after running tokenizer:')
# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
print(input_tensor.shape, target_tensor.shape)
print(input_tensor[:3])

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
print('Number of Train, Validation, and test tensors: ')
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


def convert(lang_tokenizer, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang_tokenizer.index_word[t]))


print("Input Language; index to word mapping")
convert(inp_lang_tokenizer, input_tensor_train[0])
print()
print("Target Language; index to word mapping")
convert(targ_lang_tokenizer, target_tensor_train[0])

# Create a tf.data dataset
# Need to see how padding is taken care of?
# Index of words should start from 1 and NOT zero

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024

vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1
vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print('Shape of input batch and output batches out of tf.data object')
print(example_input_batch.shape, example_target_batch.shape)

#############################################
# Write Encoder Decoder Model With Attention #
#############################################
"""
The input is put through an encoder model which gives us the encoder output of shape 
(batch_size, max_length, hidden_size)
and the encoder hidden state of shape (batch_size, hidden_size).
"""


class Encoder(keras.Model):
    """
    Todo: Make the model stack of 2 GRU layers and make sure state is passe correctly!
    """
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden_state):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden_state)
        return output, state

    def initialize_hidden_state(self):
        # Note: GRU only has hidden state!
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
# sample_input
sample_hidden_state = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden_state)
print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


class BahdanauAttention(keras.layers.Layer):
    """
    W1: (enc_units, interm_dim);    W2: (dec_units, interm_dim);    V: (interm_dim, 1);
    h_enc: (Batch_size, T, enc_units);      h_dec: (Batch_size, 1, dec_units)

    score(h_enc,  h_dec) =  tanh (h_enc * W1 + h_dec * W2)* V ; shape: (Batch_size, T, 1)
    attention_weights = softmax(score): (Batch_size, T, 1)
    context_vector = reduce_sum(attention_weights * h_enc, axis=1):  (Batch_size, enc_units)
    returns : context_vector, attention_weights
    """
    def __init__(self, intemediate_weight_mat_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(intemediate_weight_mat_dim)
        self.W2 = keras.layers.Dense(intemediate_weight_mat_dim)
        self.V = keras.layers.Dense(1)

    def call(self, query, values):
        """
        :param query: h_dec
        :param values: h_enc
        :return: context vector, attention weights
        """
        # query hidden state shape == (batch_size, hidden size) ; Decoder State!
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size) ; coming from Encoder!
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, axis=1)
        # score_shape == (batch_size, max_seq_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, unit)
        score = tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values))
        score = self.V(score)
        # Attention weights shape: (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(self.dec_units, return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = keras.layers.Dense(vocab_size)
        # used for attention:
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, decoder_state, enc_output_hidden):
        # X is one word for each sample: (batch, 1)
        # enc_output shape == (batch, max_len, hidden_size]
        context_vector, attention_weights = self.attention(decoder_state, enc_output_hidden)
        # x Shape after passing through embedding: (batch, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # I think we need to pass initial state of decoder!!!
        output, state = self.gru(x, initial_state=decoder_state)
        # passing the concatenated vector to the GRU
        # In TUTORIAL:
        # output, state = self.gru(x)

        # output shape == (batch_size, 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)
        # return attention_weights for plotting attention in evaluate!
        return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)

print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
"""
# if reduction=AUTO: will sum over batch
Y_true, Y_pred: [batch_size, d0, .. dN]
Returns:
Assume Y_pred shape:  (BATCH_SIZE, Num_Classes)
then if reduction is NOne:   returns (Batch_Size) losses

Weighted loss float Tensor. If reduction is NONE, this has shape [batch_size, d0, .. dN-1];
otherwise, it is scalar. (Note dN-1 because all loss functions reduce by 1 dimension, usually axis=-1.)
"""
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


"""
Training
1. Pass the input through the encoder which return encoder output and the encoder hidden state.
2. The encoder output, encoder hidden state and the decoder input (which is the start token) is passed to the decoder.
3. The decoder returns the predictions and the decoder hidden state.
4. The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
5. Use teacher forcing to decide the next input to the decoder.
6. Teacher forcing is the technique where the target word is passed as the next input to the decoder.
7. The final step is to calculate the gradients and apply it to the optimizer and back propagate.
"""


@tf.function
def train_step(inp, targ, enc_hidden_state):
  """Goes over a batch of sequences!!"""
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden_state = encoder(inp, enc_hidden_state)
    # At the start of translation, we warm up state of decoder by encoder hidden state
    # this requires equal number of units in Enc and decoder recurrent struct!
    dec_hidden = enc_hidden_state

    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
    print("Shape of target tensor: {}".format(targ))
    print("Shape of decoder input: {}".format(dec_input.shape))
    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      if t == 1:
        print('Shape of predictions: {}'.format(predictions.shape))

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


EPOCHS = 2

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix=checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


"""
Translate
The evaluate function is similar to the training loop, except we don't use teacher forcing here.
The input to the decoder at each time step is its previous predictions
along with the hidden state and the encoder output.
Stop predicting when the model predicts the end token.
And store the attention weights for every time step.
"""


def evaluate(sentence):
    # SINCE pad_sequences( ) PADS INPUT TO MAX_LEN_INP, Hence the second dimension of attention array
    attention_to_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        print('Shape of Attention weights reshaped for plotting: {}'.format(attention_weights.shape))
        attention_to_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang_tokenizer.index_word[predicted_id] + ' '

        if targ_lang_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention_to_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_to_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(mpl_ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl_ticker.MultipleLocator(1))
    plt.colorbar()
    plt.show()


def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  plot_attention(attention_plot, sentence.split(' '), result.split(' '))


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'hace mucho frio aqui.')


translate(u'esta es mi vida.')


translate(u'¿todavia estan en casa?')


# wrong translation
translate(u'trata de averiguarlo.')
