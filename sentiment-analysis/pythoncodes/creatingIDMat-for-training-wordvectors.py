from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from os import listdir
from os.path import isfile, join
'''
will read all the files in, create the list of unique words, create word to index dictionary,
convert each text into associated index of words.

I want to use NLTK to write 3  functions that do
1- tokenize text into list of words
2- remove punctuation (will decide on this)
3- remove stop words (choice of stop words can be decided)
using the word_list associated with the pre-trained word vectors, 
I will convert the whole corpus into (N, max_sequence_length) matrix, entry i,j represents the 
index into the word vector for word j in document i.
The data is arranged such that the first chunk is positive reviews and the second part is negative reviews.
'''
rem_stop_words = True
stop_words = list(set(stopwords.words('english')))
print('Stop word list=')
print(stop_words)
maxSeqLength = 250
num_text_files = 25000


def remove_stop_words(word_list, stop_words):
    filtered_list = [w for w in word_list if w not in stop_words]
    return filtered_list


# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
def clean_sentences(string):
    import re
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


positive_files = ['training_data/positiveReviews/' + f for f in listdir('training_data/positiveReviews')
                  if isfile(join('training_data/positiveReviews/', f))]

negative_files = ['training_data/negativeReviews/' + f for f in listdir('training_data/negativeReviews')
                  if isfile(join('training_data/negativeReviews/', f))]

# reading all files and creating the word id dictionary
# i will remove stop words and punctuation before  concatenating the text

vocab_list = list(np.loadtxt('vocabulary_list-trainingEmbedding.txt', delimiter='\t', dtype='str'))
print(vocab_list[:10])

'''
Now,  we convert each review to the list of word index to our pre-trained word vector matrix.
We'll load in the  text corpus  and integerize it to get a N x max_seq_length matrix. 
lets save the sequence length as well!
choices to make:
1. truncate sequences or not?
2. based on that sequence length should be determined!
2. need to save length of each text in order to get correct output of LSTM!
'''
vocab_size = len(vocab_list) + 1  # one extra for unknown words coming from new text
print('Size of the Vocabulary=', vocab_size)
ids = np.zeros((num_text_files, maxSeqLength), dtype='int32')
file_lengths = np.zeros(num_text_files, dtype='int32')
fileCounter = 0
k = 0
for pf in positive_files:
    if k % 100 == 0:
        print ('Number of files processed so far=',k)
    k += 1
    with open(pf, "r", encoding='UTF-8') as f:
        indexCounter = 0
        text = f.read()
        text = clean_sentences(text)
        text_tokenized = word_tokenize(text)
        if rem_stop_words:
            text_tokenized = remove_stop_words(text_tokenized, stop_words)
        curr_file_length = len(text_tokenized)
        for word in text_tokenized:
            try:
                ids[fileCounter][indexCounter] = vocab_list.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = vocab_size  # Vector for unknown words
            indexCounter += 1
            if indexCounter >= maxSeqLength:
                break
    file_lengths[fileCounter] = min(maxSeqLength, curr_file_length )
    fileCounter += 1

for nf in negative_files:
    if k % 100 == 0:
        print(k)
    k += 1
    with open(nf, "r", encoding='UTF-8') as f:
        indexCounter = 0
        text = f.read()
        text = clean_sentences(text)
        text_tokenized = word_tokenize(text)
        if rem_stop_words:
            text_tokenized = remove_stop_words(text_tokenized, stop_words)
        curr_file_length = len(text_tokenized)
        for word in text_tokenized:
            try:
                ids[fileCounter][indexCounter] = vocab_list.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = vocab_size  # Vector for unknown words
            indexCounter += 1
            if indexCounter >= maxSeqLength:
                break
    file_lengths[fileCounter] = min(maxSeqLength, curr_file_length)
    fileCounter += 1
# Pass into embedding function and see if it evaluates.
print(ids.shape)
np.save('file_lengths-training-embedding', file_lengths)
print('file lengths written to file...')
np.save('idsMatrix-training-embedding', ids)
print(ids[0])
print(file_lengths[0])



