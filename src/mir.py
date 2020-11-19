from __future__ import unicode_literals

import collections

import pandas as pd
import xml.etree.ElementTree as ET
import pickle

import src.compression as compress
import src.text_processing as proc_text
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit import print_formatted_text, HTML
import src.word_correction as wc
import threading
from pathlib import Path


class MIR:
    def __init__(self, files_root: str = './files', output_root: str = './outputs'):
        self.files_root = files_root
        self.output_root = output_root
        self.persian_wikis = []
        self.ted_talk_title = []
        self.ted_talk_desc = []
        self.lang = 'eng'
        self.dataset_loaded = False

        self.positional_indices = dict()  # key: word value: dict(): key: doc ID, value: list of positions
        self.positional_indices_title = dict()  # key: word value: dict(): key: doc ID, value: list of positions

        self.coded_indices = dict()  # key: word value: dict(): key: doc ID, value: bytes of indices
        self.bigram_indices = dict()  # key: bi-gram value: dict(): key: word, value: collection freq
        self.tokens = []
        self.collections = []
        self.collections_deleted = []  # vector indicating whether the corresponding document is deleted or not
        # creating output root
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        self.positional_add = 'outputs/pos.pickle'
        self.positional_title_add = 'outputs/pos_title.pickle'
        self.bigram_add = 'outputs/bi.pickle'
        self.coded_add = 'outputs/coded.pickle'

    def _load_talks(self, pb=None):
        talks = pd.read_csv(f'{self.files_root}/ted_talks.csv')
        self.ted_talk_title = talks['title'].to_list()
        self.ted_talk_desc = talks['description'].to_list()
        self.dataset_loaded = True
        for talk_id in pb(range(len(self.ted_talk_title)), label='Ted Talks') if pb is not None else range(
                len(self.ted_talk_title)):
            self.collections.append(
                'title: ' + self.ted_talk_title[talk_id] + '\n' + 'desc: ' + self.ted_talk_desc[talk_id])
            self.collections_deleted.append(False)
            self.insert(self.ted_talk_title[talk_id], len(self.collections) - 1, True)
            self.insert(self.ted_talk_desc[talk_id], len(self.collections) - 1)

    def _load_wikis(self, pb=None):
        root = ET.parse(f'{self.files_root}/Persian.xml').getroot()
        self.dataset_loaded = True
        for child in pb(root, label='Persian Wikis') if pb is not None else root:
            desc = ''
            title = ''
            for chil in child:
                if chil.tag[-8:] == 'revision':
                    for ch in chil:
                        if ch.tag[-4:] == 'text':
                            desc = ch.text
                elif chil.tag[-5:] == 'title':
                    title = chil.text
            self.collections.append(
                'title: ' + title + '\n' + 'desc: ' + desc)
            self.insert(title, len(self.collections) - 1, True)
            self.insert(desc, len(self.collections) - 1)

    def load_dataset(self, dataset='talks'):
        """loads datasets - dataset: ['taks'/'wikis']"""
        # resetting results
        self.persian_wikis = []
        self.ted_talk_title = []
        self.ted_talk_desc = []
        self.positional_indices = dict()
        self.coded_indices = dict()
        self.bigram_indices = dict()
        self.collections = []
        self.collections_deleted = []
        with ProgressBar(title='Loading Datasets') as pb:
            if dataset == 'talks':
                self.lang = 'eng'
                self._load_talks(pb)
            else:
                self.lang = 'persian'
                self._load_wikis(pb)

    def fix_query(self, query: str):
        """fixes queries based on the available vocabulary"""
        lang = self.lang
        dictionary = list(self.positional_indices.keys())
        fixed_query = []
        pre_query = proc_text.prepare_text(query, lang, False)
        for word in pre_query:
            if word in dictionary:
                fixed_query.append(word)
            else:
                fixed_query.append(wc.fix_word(word, dictionary))
        return ' '.join(fixed_query)

    def calc_jaccard_dist(self, word1: str, word2: str):
        """calculates the jaccard distance of two words"""
        word_list1 = [word1[i:i + 2] for i in range(len(word1) - 1)]
        word_list2 = [word2[i:i + 2] for i in range(len(word2) - 1)]
        return wc.calc_jaccard(word_list1, word_list2)

    def calc_edit_dist(self, word1, word2):
        """"calculates the edit distance of two words"""
        return wc.calc_edit_distance(word1, word2)

    def prepare_text(self, text: str, lang: str = None):
        """"""
        lang = lang or self.lang
        print(proc_text.prepare_text(text, lang, verbose=False))

    def insert(self, document, doc_id: int = None, title: bool = False):
        """insert a document"""
        lang = self.lang
        if doc_id is None:
            self.collections.append(document)
            self.collections_deleted.append(False)
            doc_id = len(self.collections) - 1
        terms = proc_text.prepare_text(document, lang, False)
        self.tokens += terms

        # Bi-gram
        words = list(set(terms))
        for word in words:
            bis = proc_text.bigram_word(word)
            for bi in bis:
                if bi not in self.bigram_indices.keys():
                    self.bigram_indices[bi] = dict()

                if word not in self.bigram_indices[bi].keys():
                    self.bigram_indices[bi][word] = 1
                else:
                    self.bigram_indices[bi][word] += 1

        # Positional
        if not title:
            for i in range(len(terms)):
                term = terms[i]
                if term not in self.positional_indices.keys():
                    self.positional_indices[term] = dict()
                if doc_id not in self.positional_indices[term].keys():
                    self.positional_indices[term][doc_id] = []
                self.positional_indices[term][doc_id].append(i)

        if title:
            for i in range(len(terms)):
                term = terms[i]
                if term not in self.positional_indices_title.keys():
                    self.positional_indices_title[term] = dict()
                if doc_id not in self.positional_indices_title[term].keys():
                    self.positional_indices_title[term][doc_id] = []
                self.positional_indices_title[term][doc_id].append(i)

    def delete(self, document, doc_id: int = None):
        lang = self.lang
        if doc_id is None:
            doc_id = self.collections.index(document)
            self.collections_deleted[doc_id] = True
        tokens = proc_text.prepare_text(document, lang)

        # Bigram
        words = list(set(tokens))
        for word in words:
            bis = proc_text.bigram_word(word)
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

    def posting_list_by_word(self, word: str):
        lang = self.lang
        # print(list(self.positional_indices.get(term, '').keys()), sep=', ')
        if not self.dataset_loaded:
            term = proc_text.prepare_text(word, lang, verbose=False)[0]
            print_formatted_text(HTML(f'<skyblue>Term:</skyblue> <cyan>{term}</cyan>'))
            print(self.positional_indices.get(term, ''))
        else:
            term = proc_text.prepare_text(word, lang, verbose=False)[0]
            print_formatted_text(HTML(f'<skyblue>Term:</skyblue> <cyan>{term}</cyan>'))
            print_formatted_text(HTML(f'<red>Title:</red>'))
            print(self.positional_indices_title.get(term, ''))
            print_formatted_text(HTML(f'<red>Description:</red>'))
            print(self.positional_indices.get(term, ''))

    def words_by_bigram(self, bigram: str):
        """get all possible words containing the given bigram"""
        print(*list(self.bigram_indices.get(bigram, dict()).keys()), sep=', ')

    def words_by_bigram_suggestion(self, args):
        if not args or not args[0]:
            return self.bigram_indices.keys()
        if len(args) > 1:
            return []
        return filter(lambda x: x.startswith(args[0]), self.bigram_indices.keys())

    def save_indices(self):
        with open(self.positional_add, 'wb') as handle:
            pickle.dump(self.positional_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.positional_title_add, 'wb') as handle:
            pickle.dump(self.positional_indices_title, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.bigram_add, 'wb') as handle:
            pickle.dump(self.bigram_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_indices(self):
        with open(self.positional_add, 'rb') as handle:
            self.positional_indices = pickle.load(handle)
        with open(self.positional_title_add, 'rb') as handle:
            self.positional_indices_title = pickle.load(handle)
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)

    def save_coded_indices(self):  # todo: arvin bitarray
        self._code_indices()
        with open(self.coded_add, 'wb') as handle:
            pickle.dump(self.coded_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.bigram_add, 'wb') as handle:
            pickle.dump(self.bigram_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_coded_indices(self):  # todo: arvin bitarray
        with open(self.coded_add, 'rb') as handle:
            self.coded_indices = pickle.load(handle)
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)
        self._decode_indices()

    def _code_indices(self, coding: str = "s"):  # todo: arvin bitarray
        for word in self.positional_indices:
            self.coded_indices[word] = dict()
            for doc in self.positional_indices[word]:
                self.coded_indices[word][doc] = compress.gamma_code(
                    self.positional_indices[word][doc]) if coding == "gamma" else compress.variable_byte(
                    self.positional_indices[word][doc])

    def _decode_indices(self, coding: str = "s"):  # todo: arvin bitarray
        for word in self.coded_indices:
            self.positional_indices[word] = dict()
            for doc in self.coded_indices[word]:
                self.positional_indices[word][doc] = compress.decode_gamma_code(format(
                    self.coded_indices[word][doc], "b")) if coding == "gamma" else compress.decode_variable_length(
                    format(self.coded_indices[word][doc], "b"))

    def sort_by_relevance(self, query: str):
        lang = self.lang
        query = proc_text.prepare_text(query, lang, False)
        for item in query:
            print(item, item in self.positional_indices)

    def find_stop_words(self):

        counter = collections.Counter(self.tokens)
        word_freq = sum(counter.values())
        stop_words = []
        for key in counter.keys():
            if counter[key] / word_freq >= 0.005:
                stop_words.append(key)
        print(stop_words)
