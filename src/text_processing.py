import collections
import nltk
import string
from nltk.stem import PorterStemmer
import hazm


def prepare_text(text, lang, verbose=True):
    # todo: return only terms (tokenized + stem + lemmatized)
    if lang == 'persian':
        punctuation_marks = ['!', '؟', '،', '.', '؛', ':', '«', '»','<', '>', '-', '[', ']', '{', '}', '|', ')', '(', '/',
                             '=', '*', '\'']
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


def bigram_word(word):
    bis = []
    for i in range(len(word) - 1):
        bis.append(word[i:i + 2])
    bis.append('$' + word[0])
    bis.append(word[-1] + '$')
    return bis
