from __future__ import unicode_literals
from hazm import *
import pandas as pd
import xml.etree.ElementTree as ET
import collections
import nltk
import string
from nltk.stem import PorterStemmer
import pickle


class MIR:
    def __init__(self):
        self.persian_wikis = []
        self.ted_talk_title = []
        self.ted_talk_desc = []
        self.positional_indices = dict()  # key: word value: dict(): key: doc ID, value: list of positions
        self.bigram_indices = dict()  # key: bi-gram value: dict(): key: word, value: collection freq
        self.collections = []
        self.collections_deleted = []  # vector indicating whether the corresponding document is deleted or not

        self.positional_add = 'src/pos.pickle'
        self.bigram_add = 'src/bi.pickle'

    def load_datasets(self, dataset='talks'):
        # Loading Ted Talks
        if dataset == 'talks':
            talks = pd.read_csv('src/ted_talks.csv')
            self.ted_talk_title = talks['title'].to_list()
            self.ted_talk_desc = talks['description'].to_list()
            for talk_id in range(len(self.ted_talk_title)):
                self.collections.append(
                    'title: ' + self.ted_talk_title[talk_id] + '\n' + 'desc: ' + self.ted_talk_desc[talk_id])
                self.collections_deleted.append(False)
                self.insert(self.ted_talk_title[talk_id], 'eng', len(self.collections) - 1)
                self.insert(self.ted_talk_desc[talk_id], 'eng', len(self.collections) - 1)

        # Loading Wiki pages into a list containing their text
        elif dataset == 'wikis':
            root = ET.parse('src/Persian.xml').getroot()
            for child in root:
                for chil in child:
                    if chil.tag[-8:] == 'revision':
                        for ch in chil:
                            if ch.tag[-4:] == 'text':
                                self.persian_wikis.append(ch.text)
            for wiki in self.persian_wikis:
                self.insert(wiki, 'persian')

    @staticmethod
    def bigram_word(word):
        bis = []
        for i in range(len(word) - 1):
            bis.append(word[i:i + 2])
        bis.append('$' + word[0])
        bis.append(word[-1] + '$')
        return bis

    def insert(self, document, lang, doc_id=None):
        if doc_id is None:
            self.collections.append(document)
            self.collections_deleted.append(False)
            doc_id = len(self.collections) - 1
        tokens = self.prepare_text(document, lang)

        # Bigram
        words = list(set(tokens))
        for word in words:
            bis = self.bigram_word(word)
            for bi in bis:
                if bi not in self.bigram_indices.keys():
                    self.bigram_indices[bi] = dict()

                if word not in self.bigram_indices[bi].keys():
                    self.bigram_indices[bi][word] = 1
                else:
                    self.bigram_indices[bi][word] += 1

        # Positional
        for i in range(len(tokens)):
            token = tokens[i]
            if token not in self.positional_indices.keys():
                self.positional_indices[token] = dict()
            if doc_id not in self.positional_indices[token].keys():
                self.positional_indices[token][doc_id] = []
            self.positional_indices[token][doc_id].append(i)

    def delete(self, document, lang, doc_id=None):
        if doc_id is None:
            doc_id = self.collections.index(document)
            self.collections_deleted[doc_id] = True
        tokens = self.prepare_text(document, lang)

        # Bigram
        words = list(set(tokens))
        for word in words:
            bis = self.bigram_word(word)
            for bi in bis:
                self.bigram_indices[bi][word] -= 1
                if self.bigram_indices[bi][word] == 0:
                    del self.bigram_indices[bi][word]
                if len(self.bigram_indices[bi].keys()) == 0:
                    del self.bigram_indices[bi]

        # Positional
        keys_to_del = []
        for key in self.positional_indices.keys():
            if doc_id in self.positional_indices[key].keys():
                del self.positional_indices[key][doc_id]
            if len(self.positional_indices[key].keys()) == 0:
                keys_to_del.append(key)
        for kdl in keys_to_del:
            del self.positional_indices[kdl]

    def posting_list_by_word(self,word,lang):
        token=self.prepare_text(word,lang,verbose=False)[0]
        print(self.positional_indices[token])

    def words_by_bigram(self,bigram):
        print(self.bigram_indices[bigram].keys())

    def save_indices(self):
        with open(self.positional_add, 'wb') as handle:
            pickle.dump(self.positional_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.bigram_add, 'wb') as handle:
            pickle.dump(self.bigram_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_indices(self):
        with open(self.positional_add, 'rb') as handle:
            self.positional_indices = pickle.load(handle)
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)

    @staticmethod
    def prepare_text(text, lang,verbose=True):
        if lang == 'persian':
            punctuation_marks = ['!', '؟', '،', '.', '؛', ':', '«', '»', '-', '[', ']', '{', '}', '|', ')', '(', '/',
                                 '=', '*', '\'']
            for punc in punctuation_marks:
                text = text.replace(punc, " ")
            normalizer = Normalizer()
            normalized_text = normalizer.normalize(text)
            tokens = word_tokenize(normalized_text)
            stemmer = Stemmer()
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


mir = MIR()
mir.load_datasets('talks')
