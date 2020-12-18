import collections
import nltk
import string
from nltk.stem import PorterStemmer
import hazm
import numpy as np


def prepare_text(text, lang='eng', verbose=False):
    if lang == 'persian':
        punctuation_marks = ['!', '؟', '،', '.', '؛', ':', '«', '»', '<', '>', '-', '[', ']', '{', '}', '|', ')', '(',
                             '/', '=', '*', '\'', ',', '"', '`', '?']
        for punc in punctuation_marks:
            text = text.replace(punc, " ")
        normalizer = hazm.Normalizer()
        normalized_text = normalizer.normalize(text)
        tokens = hazm.word_tokenize(normalized_text)
        stemmer = hazm.Stemmer()
        clean_tokens = []
        for i in range(len(tokens)):
            tokens[i] = stemmer.stem(tokens[i])
        for token in tokens:
            if len(token) != 0:
                clean_tokens.append(token)
        if verbose:
            print('Tokens:', clean_tokens)
        return clean_tokens
    if lang == 'eng':
        # Lowering
        text = text.lower()
        # removing puncs
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
        text = text.translate(translator)

        tokens = nltk.word_tokenize(text)

        porter = PorterStemmer()
        clean_tokens = []
        for i in range(len(tokens)):
            tokens[i] = porter.stem(tokens[i])
        for token in tokens:
            if len(token) != 0:
                clean_tokens.append(token)
        if verbose:
            print('Tokens:', clean_tokens)
        return clean_tokens


def vocab(docs):
    vocab_dict = dict()
    for doc in docs:
        for i in doc:
            if i not in vocab_dict:
                vocab_dict[i] = 0
            vocab_dict[i] += 1
    idxs = sorted(vocab_dict)
    for i, j in enumerate(idxs):
        vocab_dict[j] = i, vocab_dict[j]
    return vocab_dict


def bigram_word(word):
    bis = []
    for i in range(len(word) - 1):
        bis.append(word[i:i + 2])
    bis.append('$' + word[0])
    bis.append(word[-1] + '$')
    return bis
