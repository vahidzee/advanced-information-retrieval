import collections
import nltk
import string
from nltk.stem import PorterStemmer
import hazm
import numpy as np

hazm_normalizer = hazm.Normalizer()
hazm_stemmer = hazm.Stemmer()
hazm_lemmatizer = hazm.Lemmatizer()

persian_puncts = '\u060C\u061B\u061F\u0640\u066A\u066B\u066C'
persian_numerics = '\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9'
english_numerics = '0123456789'
chars_to_remove = string.punctuation + persian_puncts + persian_numerics + english_numerics
persian_conjuction = {'از', 'به', 'با', 'بر', 'برای', 'در', 'و', 'که', 'را'}
persian_translator = str.maketrans('', '', chars_to_remove)


def persian_terms(text, stem=False, lemmatize=False, remove_conjunctions=False, join=None):
    normalized_text = hazm_normalizer.normalize(text.translate(persian_translator))
    result = hazm.word_tokenize(normalized_text)
    if stem:
        result = [hazm_stemmer.stem(x) for x in result]
    if lemmatize:
        result = [hazm_lemmatizer.lemmatize(x) for x in result]
    if remove_conjunctions:
        result = [x for x in result if x not in persian_conjuction]
    if join is not None:
        return join.join(result)
    return result


def prepare_text(text, lang='eng', verbose=False):
    if lang == 'persian':
        clean_tokens = persian_terms(text, False, True, True)
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
