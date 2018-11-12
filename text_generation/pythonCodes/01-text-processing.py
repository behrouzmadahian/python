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
data_dir = 'C:/behrouz/PythonCodes/ML/text_generation/'


def remove_stop_words(word_list, stwords):
    filtered_list = [w for w in word_list if w not in stwords]
    return filtered_list


# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
def clean_sentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def data_process(word_list, vocab_list, seq_len):
    word_ind = [vocab_list.index(w) for w in word_list]
    datax, datay = [], []
    for i in range(0, len(word_ind) - seq_len):
        x = word_ind[i: i + seq_len]
        y = word_ind[i + seq_len]
        datax.append(x)
        datay.append(y)
    return np.array(datax).astype(int), np.array(datay).astype(int)


with open(data_dir + 'Alice_in_wonderland.txt', encoding='UTF-8') as f:
    text = f.read()
    cleaned_text = clean_sentences(text)
    text_tokenized = word_tokenize(cleaned_text)
print(len(text_tokenized))
print(text_tokenized[:10])
vocab_list = list(set(text_tokenized))
print('Length of vocabulary: ', len(vocab_list))
np.savetxt(data_dir + 'vocabulary_list.txt', vocab_list, delimiter='\t', fmt='%s')

datax, datay = data_process(text_tokenized, vocab_list, 30)
np.save(data_dir + 'datax.npy', datax)
np.save(data_dir + 'datay.npy', datay)