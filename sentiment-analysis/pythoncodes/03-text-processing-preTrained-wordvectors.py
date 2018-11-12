from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from matplotlib import pyplot as plt
'''
I want to use NLTK to write 3  functions that do
1- tokenize text into list of words
2- remove punctuation (will decide on this)
3- remove stop words (choice of stop words can be decided)
using the word_list associated with the pre-trained word vectors, 
I will convert the whole corpus into (N, max_sequence_length) matrix, entry i,j represents the 
index into the word vector for word j in document i.
The data is arranged such that the first chunk is positive reviews and the second part is negative reviews.
'''
data_dir = 'C:/behrouz/PythonCodes/ML/sentiment-analysis/training_data/'


def remove_stop_words(word_list, stop_words):
    filtered_list = [w for w in word_list if w not in stop_words]
    return filtered_list


# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
def clean_sentences(string):
    import re
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


#  wordsList = np.load('training_data/wordsList.npy')
wordsList = np.loadtxt(data_dir + 'vocabulary_list-trainingEmbedding.txt', dtype=str)
print(wordsList.shape)
print('shape of list of ALL words in corpus=', wordsList.shape)
wordsList = wordsList.tolist()
print('first 10 words in our wordslist=(binary format)', wordsList[:10])
# encode words as UTF-8
# wordsList = [word.decode('UTF-8') for word in wordsList] # use only if loading from binary file!
print('first 10 words in our wordslist=(UTF-8)', wordsList[:10])
# loading word vectors:
wordVectors = np.load(data_dir+'wordVectors.npy')
print(wordVectors.shape)
print('Loaded the word vectors!')

from os import listdir
from os.path import isfile, join
rem_stop_words = True
stop_words = list(set(stopwords.words('english')))
print(stop_words)


positive_files = [data_dir+'/positiveReviews/' + f for f in listdir(data_dir + '/positiveReviews/')
                  if isfile(join(data_dir+'/positiveReviews/', f))]

negative_files = [data_dir+'/negativeReviews/' +  f for f in listdir(data_dir + '/negativeReviews')
                  if isfile(join(data_dir + '/negativeReviews', f))]
print('Total positive reviews=', len(positive_files))
print('Total negative reviews=', len(negative_files))
# calculating statistics of whole corpus to decide on max_seq_length:
# we can choose max sequence length to be the length of document with maximum length
numWords = []
for pf in positive_files:
    with open(pf, 'r', encoding='UTF-8') as f:
        text = f.read()
        text = clean_sentences(text)
        text_tokenized = word_tokenize(text)
        if rem_stop_words:
            text_tokenized = remove_stop_words(text_tokenized, stop_words)
    numWords.append(len(text_tokenized))
print('Total number of words in positive texts=', sum(numWords))

for nf in negative_files:
    with open(nf, 'r', encoding='UTF-8') as f:
        text = f.read()
        text = clean_sentences(text)
        text_tokenized = word_tokenize(text)
        if rem_stop_words:
            text_tokenized = remove_stop_words(text_tokenized, stop_words)
    numWords.append(len(text_tokenized))
numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

 #histogram of data
plt.hist(numWords, bins=50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()
'''
From the histogram as well as the average number of words per file, we can safely say 
that most reviews will fall under 250 words, which is the max sequence length value we will set.
'''
maxSeqLength = 250

'''
Now,  we convert each review to the list of word index to our pre-trained word vector matrix.
We'll load in the  text corpus  and integerize it to get a N x max_seq_length matrix. 
lets save the sequence length as well!
choices to make:
1. truncate sequences or not?
2. based on that, sequence length should be determined!
3. need to save length of each text in order to get correct output of LSTM!
'''

ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
file_lengths = np.zeros(numFiles, dtype='int32')
fileCounter = 0
for pf in positive_files:
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
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 399999  # Vector for unknown words
            indexCounter += 1
            if indexCounter >= maxSeqLength:
                break
    file_lengths[fileCounter] = min(maxSeqLength, curr_file_length)
    fileCounter += 1

for nf in negative_files:
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
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 399999  # Vector for unknown words
            indexCounter += 1
            if indexCounter >= maxSeqLength:
                break
    file_lengths[fileCounter] = min(maxSeqLength, curr_file_length)
    fileCounter = fileCounter + 1
# Pass into embedding function and see if it evaluates.
print(ids.shape)
np.save(data_dir+'idsMatrix', ids)
np.save(data_dir+'file_lengths', file_lengths)
print(ids[0])
print(file_lengths[0])

