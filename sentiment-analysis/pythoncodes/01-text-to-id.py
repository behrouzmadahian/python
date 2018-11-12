import re
from nltk.corpus import stopwords
from nltk import word_tokenize
stop_words = stopwords.words('english')


def remove_stop_words(word_list, stopwords):
    filtered_list = [w for w in word_list if not w in stopwords]
    return filtered_list


# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
def clean_sentences(string):
    strip_special_chars = re.compile("[^a-zA-Z0-9_]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def text_to_ids(text, vocab_list):
    text_cleaned = clean_sentences(text)
    word_list = word_tokenize(text_cleaned)
    word_list = remove_stop_words(word_list, stop_words)
    word_inds = [vocab_list.index(w) for w in word_list]
    return word_inds
