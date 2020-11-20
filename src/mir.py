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
from src.vector_space import score_query, scale_lnc, logarithmic
from pathlib import Path
from src.utils import print_match_doc
import os


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
        self.coded_title_indices = dict()
        self.vocabulary = set()
        self.bigram_indices = dict()  # key: bi-gram value: dict(): key: word, value: collection freq
        self.collections = []
        self.collections_deleted = []  # vector indicating whether the corresponding document is deleted or not

        # creating output root
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        self.positional_add = 'outputs/pos.pickle'
        self.positional_title_add = 'outputs/pos_title.pickle'
        self.bigram_add = 'outputs/bi.pickle'
        self.coded_add = 'outputs/coded.pickle'
        self.coded_title_add = 'outputs/coded_titles.pickle'

    # part 0
    def _load_talks(self, pb=None):
        talks = pd.read_csv(f'{self.files_root}/ted_talks.csv')
        self.ted_talk_title = talks['title'].to_list()
        self.ted_talk_desc = talks['description'].to_list()
        for talk_id in pb(range(len(self.ted_talk_title)), label='Ted Talks') if pb is not None else range(
                len(self.ted_talk_title)):
            self._insert(self.ted_talk_title[talk_id], self.ted_talk_desc[talk_id])

    def _load_wikis(self, pb=None):
        root = ET.parse(f'{self.files_root}/Persian.xml').getroot()
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
            self._insert(title, desc)

    def load_dataset(self, dataset='talks'):
        """loads a dataset - dataset: ['taks'/'wikis']"""
        # resetting results
        self.persian_wikis = []
        self.ted_talk_title = []
        self.ted_talk_desc = []
        self.positional_indices = dict()
        self.positional_indices_title = dict()
        self.coded_indices = dict()
        self.coded_title_indices = dict()
        self.bigram_indices = dict()
        self.collections = []
        self.collections_deleted = []
        self.dataset_loaded = True
        self.vocabulary = set()
        with ProgressBar(title='Loading Datasets') as pb:
            if dataset == 'talks':
                self.lang = 'eng'
                self._load_talks(pb)
            else:
                self.lang = 'persian'
                self._load_wikis(pb)

    def load_dataset_suggestion(self, args):
        choices = ['talks', 'wikis']
        if not args or not args[0]:
            return choices
        if len(args) > 1:
            return []
        return filter(lambda x: x.startswith(args[0]), choices)

    # part 1
    def prepare_text(self, text: str, lang: str = None):
        """pre-processes text based on the specified language"""
        lang = lang or self.lang
        terms = proc_text.prepare_text(text, lang, verbose=False)
        print_formatted_text(HTML(f'<skyblue>Terms:</skyblue> <i>{" ".join(terms)}</i>'))

    def prepare_text_suggestion(self, args):
        choices = ['eng', 'persian', 'none']
        if len(args) > 1:
            return choices
        return []

    def _insert_bigram(self, word):
        bis = proc_text.bigram_word(word)
        for bi in bis:
            if bi not in self.bigram_indices.keys():
                self.bigram_indices[bi] = dict()

            if word not in self.bigram_indices[bi].keys():
                self.bigram_indices[bi][word] = 1
            else:
                self.bigram_indices[bi][word] += 1

    def _insert_position(self, terms, dictionary, doc_id):
        for i in range(len(terms)):
            term = terms[i]
            if term not in dictionary.keys():
                dictionary[term] = dict()
            if doc_id not in dictionary[term].keys():
                dictionary[term][doc_id] = []
            dictionary[term][doc_id].append(i)
            self.vocabulary.add(term)

    def _insert(self, title, description):
        lang = self.lang
        if self.collections_deleted:
            doc_id = self.collections_deleted.pop(0)
            self.collections[doc_id] = (title, description)
        else:
            self.collections.append((title, description))
            doc_id = len(self.collections) - 1
        terms_document = proc_text.prepare_text(description, lang, False)
        terms_title = proc_text.prepare_text(title, lang, False)

        # Bi-gram
        for word in set(terms_document):
            self._insert_bigram(word)
        for word in set(terms_title):
            self._insert_bigram(word)

        # Positional
        self._insert_position(terms_document, self.positional_indices, doc_id)
        self._insert_position(terms_title, self.positional_indices_title, doc_id)
        return doc_id

    def insert(self, title: str, description: str):
        """inserts a document into the collection - title: document title, description: document's description"""
        doc_id = self._insert(title, description)
        print_match_doc(doc_id=doc_id, title=title, description=description)

    @staticmethod
    def _delete_position(doc_id, dictionary):
        keys_to_del = []
        for key in dictionary.keys():
            if doc_id in dictionary[key].keys():
                del dictionary[key][doc_id]
            if len(dictionary[key].keys()) == 0:
                keys_to_del.append(key)
        for kdl in keys_to_del:
            del dictionary[kdl]

    def delete(self, doc_id: int):
        """deletes a document from the collection - doc_id: document's identifier"""
        lang = self.lang
        if doc_id > len(self.collections) or doc_id in self.collections_deleted or doc_id < 0:
            return

        terms_description = proc_text.prepare_text(self.collections[doc_id][1], lang)
        terms_title = proc_text.prepare_text(self.collections[doc_id][0], lang)

        # Positionals
        self._delete_position(doc_id, self.positional_indices_title)
        self._delete_position(doc_id, self.positional_indices)

        words = list(set(terms_description + terms_title))
        for word in words:
            # vocabulary
            if word not in self.positional_indices and word not in self.positional_indices_title:
                self.vocabulary.remove(word)

            # bigram
            bis = proc_text.bigram_word(word)
            for bi in bis:
                self.bigram_indices[bi][word] -= 1
                if self.bigram_indices[bi][word] == 0:
                    del self.bigram_indices[bi][word]
                if len(self.bigram_indices[bi].keys()) == 0:
                    del self.bigram_indices[bi]

        self.collections_deleted.append(doc_id)
        print('Document', doc_id, 'deleted successfully')

    def find_stop_words(self):
        counter = collections.Counter(self.vocabulary)
        word_freq = sum(counter.values())
        stop_words = []
        for key in counter.keys():
            if counter[key] / word_freq >= 0.005:
                stop_words.append(key)
        print(stop_words)

    # part 2
    def posting_list_by_word(self, word: str):
        lang = self.lang
        if not self.dataset_loaded:
            self.prepare_text(word, lang)
            return
        term = proc_text.prepare_text(word, lang, verbose=False)[0]
        print_formatted_text(HTML(f'<skyblue>Term:</skyblue> <cyan>{term}</cyan>'))
        for idx, values in self.positional_indices_title.get(term, dict()).items():
            print_match_doc(
                idx, positions_title=values, title=self.collections[idx][0], terms=[term],
                print_terms=False, lang=lang,
                description=None if idx not in self.positional_indices.get(term, dict()) else self.collections[idx][1],
                positions_description=self.positional_indices.get(term, dict()).get(idx, tuple())
            )
        for idx, values in self.positional_indices.get(term, dict()).items():
            if idx in self.positional_indices_title.get(term, set()):
                continue
            print_match_doc(idx, positions_description=values, description=self.collections[idx][1], terms=[term],
                            print_terms=False, lang=lang)

    def posting_list_by_word_suggestion(self, args):
        if not args or not args[0]:
            return self.vocabulary
        if len(args) > 1:
            return []
        return filter(lambda x: x.startswith(args[0]), self.vocabulary)

    def words_by_bigram(self, bigram: str):
        """get all possible words containing the given bigram"""
        print(*list(self.bigram_indices.get(bigram, dict()).keys()), sep='\n')

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
        print("positional indices size:", os.stat(self.positional_add).st_size)
        print("positional titles indices size:", os.stat(self.positional_title_add).st_size)
        print("bigram indices size:", os.stat(self.bigram_add).st_size)

    def load_indices(self):
        with open(self.positional_add, 'rb') as handle:
            self.positional_indices = pickle.load(handle)
        with open(self.positional_title_add, 'rb') as handle:
            self.positional_indices_title = pickle.load(handle)
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)

    # part 3
    def save_coded_indices(self, coding: str = "gamma"):
        """compress and encode and save the indices - coding: ['gamma', 'variable']"""
        self._code_indices(coding)
        with open(self.coded_add, 'wb') as handle:
            pickle.dump(self.coded_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.bigram_add, 'wb') as handle:
            pickle.dump(self.bigram_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.coded_title_add, 'wb') as handle:
            pickle.dump(self.coded_title_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("coding:", coding)
        print("coded positional indices size:", os.stat(self.coded_add).st_size)
        print("coded positional titles indices size:", os.stat(self.coded_title_add).st_size)
        print("bigram indices size:", os.stat(self.bigram_add).st_size)

    def load_coded_indices(self, coding: str = "gamma"):
        """load the saved coded indices - coding: ['gamma', 'variable']"""
        with open(self.coded_add, 'rb') as handle:
            self.coded_indices = pickle.load(handle)
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)
        with open(self.coded_title_add, 'rb') as handle:
            self.coded_title_indices = pickle.load(handle)
        self._decode_indices(coding)

    def load_coded_indices_suggestion(self, args):
        codings = ['gamma', 'variable']
        if not args or not args[0]:
            return codings
        if len(args) > 1:
            return []
        return filter(lambda x: x.startswith(args[0]), codings)

    def save_coded_indices_suggestion(self, args):
        return self.load_coded_indices_suggestion(args)

    def _code_indices(self, coding: str = "variable"):
        for word in self.positional_indices:
            self.coded_indices[word] = dict()
            for doc in self.positional_indices[word]:
                self.coded_indices[word][doc] = compress.gamma_code(
                    self.positional_indices[word][doc]) if coding == "gamma" else compress.variable_byte(
                    self.positional_indices[word][doc])
        for word in self.positional_indices_title:
            self.coded_title_indices[word] = dict()
            for doc in self.positional_indices_title[word]:
                self.coded_title_indices[word][doc] = compress.gamma_code(
                    self.positional_indices_title[word][doc]) if coding == "gamma" else compress.variable_byte(
                    self.positional_indices_title[word][doc])

    def _decode_indices(self, coding: str = "variable"):
        for word in self.coded_indices:
            self.positional_indices[word] = dict()
            for doc in self.coded_indices[word]:
                self.positional_indices[word][doc] = compress.decode_gamma_code(format(
                    self.coded_indices[word][doc], "b")) if coding == "gamma" else compress.decode_variable_length(
                    format(self.coded_indices[word][doc], "b"))
        for word in self.coded_title_indices:
            self.positional_indices_title[word] = dict()
            for doc in self.coded_title_indices[word]:
                self.positional_indices_title[word][doc] = compress.decode_gamma_code(format(
                    self.coded_title_indices[word][doc],
                    "b")) if coding == "gamma" else compress.decode_variable_length(
                    format(self.coded_title_indices[word][doc], "b"))

    def fix_query(self, query: str):
        """fixes queries based on the available vocabulary"""
        lang = self.lang
        dictionary = list(self.positional_indices.keys()) + list(self.positional_indices_title.keys())
        fixed_query = []
        pre_query = proc_text.prepare_text(query, lang, False)
        for word in pre_query:
            if word in dictionary:
                fixed_query.append(word)
            else:
                fixed_query.append(wc.fix_word(word, dictionary))
        return ' '.join(fixed_query)

    # part 4
    def calc_jaccard_dist(self, word1: str, word2: str):
        """calculates the jaccard distance of two words"""
        word_list1 = [word1[i:i + 2] for i in range(len(word1) - 1)]
        word_list2 = [word2[i:i + 2] for i in range(len(word2) - 1)]
        return wc.calc_jaccard(word_list1, word_list2)

    def calc_edit_dist(self, word1: str, word2: str):
        """calculates the edit distance of two words"""
        return wc.calc_edit_distance(word1, word2)

    # part 5
    def _score_docs(self, query_terms, zone='title'):
        lang = self.lang
        dictionary = self.positional_indices if zone == 'description' else self.positional_indices_title
        collection_idx = 1 if zone == 'description' else 0
        query_scores = score_query(query_terms, dictionary, len(self.collections))
        result = dict()
        for term, term_score in zip(query_terms, query_scores):
            posting = dictionary.get(term, dict())
            for doc, positions in posting.items():
                if doc not in result:
                    result[doc] = 0
                result[doc] += logarithmic(len(positions)) * term_score

        for doc in result:
            result[doc] /= scale_lnc(self.collections[doc][collection_idx], lang)
        return result

    def sort_by_relevance(self, query: str, k: int = 10):
        """find top-k relevant documents based on search"""
        lang = self.lang
        self.prepare_text(query, lang)  # print the query terms
        query_terms = proc_text.prepare_text(query, lang, False)
        title_scores = self._score_docs(query_terms, 'title')
        description_scores = self._score_docs(query_terms, 'description')
        result = dict()
        for doc in description_scores:
            result[doc] = description_scores[doc] + title_scores.get(doc, 0)
        for doc_id, score in sorted(list(result.items())[:k], key=lambda x: x[1], reverse=True):
            print_match_doc(
                doc_id=doc_id, score=score, description=self.collections[doc_id][1], title=self.collections[doc_id][0],
                terms=query_terms, print_terms=False
            )

    def proximity_search(self, query: str, window: int = 5):
        pass
