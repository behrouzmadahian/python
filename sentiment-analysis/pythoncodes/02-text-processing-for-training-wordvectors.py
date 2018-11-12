from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import re
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
print(stop_words)
data_dir = 'C:/behrouz/PythonCodes/ML/sentiment-analysis/training_data/'


def remove_stop_words(word_list, stwords):
    filtered_list = [w for w in word_list if w not in stwords]
    return filtered_list


# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
def clean_sentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


positive_files = [data_dir + 'positiveReviews/' + f for f in listdir(data_dir + 'positiveReviews')
                  if isfile(join(data_dir+'positiveReviews/', f))]

negative_files = [data_dir + 'negativeReviews/' + f for f in listdir(data_dir + 'negativeReviews')
                  if isfile(join(data_dir + 'negativeReviews/', f))]
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

# histogram of data
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

# reading all files and creating the word id dictionary
# I will remove stop words and punctuation before  concatenating the text
vocab_list = []
for pf in positive_files:
    with open(pf, "r", encoding='UTF-8') as f:
        text = f.read()
        cleaned_text = clean_sentences(text)
        cleaned_text_tokenized = word_tokenize(cleaned_text)
        if rem_stop_words:
            cleaned_text_tokenized = remove_stop_words(cleaned_text_tokenized, stop_words)
    vocab_list.extend(cleaned_text_tokenized)

for nf in negative_files:
    curr_file_length = 0
    with open(nf, "r", encoding='UTF-8') as f:
        text = f.read()
        cleaned_text = clean_sentences(text)
        cleaned_text_tokenized = word_tokenize(cleaned_text)
        if rem_stop_words:
            cleaned_text_tokenized = remove_stop_words(cleaned_text_tokenized, stop_words)
    vocab_list.extend(cleaned_text_tokenized)

print(len(vocab_list))
vocab_list = list(set(vocab_list))
print(len(vocab_list))
print(vocab_list[:10])
np.savetxt(data_dir + 'vocabulary_list-trainingEmbedding.txt', vocab_list, delimiter='\t', fmt='%s')
