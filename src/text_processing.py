import collections
import nltk
import string
from nltk.stem import PorterStemmer
import hazm


def prepare_text(text, lang, verbose=True):
    # todo: return only terms (tokenized + stem + lemmatized)
    if lang == 'persian':
        punctuation_marks = ['!', '؟', '،', '.', '؛', ':', '«', '»', '-', '[', ']', '{', '}', '|', ')', '(', '/',
                             '=', '*', '\'']
        for punc in punctuation_marks:
            text = text.replace(punc, " ")
        normalizer = hazm.Normalizer()
        normalized_text = normalizer.normalize(text)
        tokens = hazm.word_tokenize(normalized_text)
        stemmer = hazm.Stemmer()
        counter = collections.Counter(tokens)
        word_freq = sum(counter.values())
        stop_words = []
        for key in counter.keys():
            if counter[key] / word_freq >= 0.3:
                stop_words.append(key)
        if verbose:
            print('Stop Words:', stop_words)
        clean_tokens = [token for token in tokens if token not in stop_words]
        for i in range(len(clean_tokens)):
            clean_tokens[i] = stemmer.stem(clean_tokens[i])
        clean_tokens = [token for token in clean_tokens if len(token) != 0]
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

        counter = collections.Counter(tokens)
        word_freq = sum(counter.values())
        stop_words = []
        porter = PorterStemmer()
        for key in counter.keys():
            if counter[key] / word_freq >= 1.6:
                stop_words.append(key)
        if verbose:
            print('Stop Words:', stop_words)
        clean_tokens = [token for token in tokens if token not in stop_words]
        for i in range(len(clean_tokens)):
            clean_tokens[i] = porter.stem(clean_tokens[i])
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

