from __future__ import unicode_literals

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import pickle
import warnings
import src.compression as compress
import src.text_processing as proc_text
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit import print_formatted_text, HTML
import src.word_correction as wc
from src.vector_space import score_query, scale_lnc, logarithmic, ntn_vectorize
from pathlib import Path
from src.utils import print_match_doc, print_evaluation_results, mix_evaluation_results
import src.classifiers as classifiers
import os

warnings.filterwarnings('ignore')
RANDOM_SEED = 666


class MIR:
    def __init__(self, files_root: str = './files', output_root: str = './outputs'):
        self.files_root = files_root
        self.output_root = output_root
        self.ted_talk_title = []
        self.ted_talk_desc = []
        self.lang = 'eng'
        self.dataset_loaded = False

        self.positional_indices = dict()  # key: word value: dict(): key: doc ID, value: list of positions
        self.positional_indices_title = dict()  # key: word value: dict(): key: doc ID, value: list of positions

        self.coded_indices = dict()  # key: word value: dict(): key: doc ID, value: bytes of indices
        self.coded_title_indices = dict()
        self.vocabulary = dict()
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

        # phase2
        self.train_term_mapping = None  # vocabulary out of train & val data (dict of <term: sorted_idx>)
        self.test_term_mapping = None  # vocabulary mapping out of test  (dict of <term: sorted_idx>)
        self.talks_term_mapping = None  # vocabulary mapping out of test  (dict of <term: sorted_idx>)
        self.train_ratio = 0.9  # how much of the train.csv should be accounted for training
        self.train_vectors = None  # ntn_vectors , views
        self.val_vectors = None  # ntn_vectors , views
        self.test_vectors = None
        self.talks_vectors = None
        self.models = list()
        self.best_model = None

    # part 0
    def _load_talks(self, pb=None):
        talks = pd.read_csv(f'{self.files_root}/talks.csv')
        ted_talk_title = talks['title'].to_list()
        ted_talk_desc = talks['description'].to_list()
        for talk_id in pb(range(len(ted_talk_title)), label='Ted Talks') if pb is not None else range(
                len(ted_talk_title)):
            self._insert(ted_talk_title[talk_id], ted_talk_desc[talk_id])

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
        self.positional_indices = dict()
        self.positional_indices_title = dict()
        self.coded_indices = dict()
        self.coded_title_indices = dict()
        self.bigram_indices = dict()
        self.collections = []
        self.collections_deleted = []
        self.dataset_loaded = True
        self.vocabulary = dict()
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
            if term not in self.vocabulary:
                self.vocabulary[term] = 0
            self.vocabulary[term] += 1

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

    # todo: phase2?
    def insert(self, title: str, description: str):
        """inserts a document into the collection - title: document title, description: document's description"""
        doc_id = self._insert(title, description)
        print_match_doc(doc_id=doc_id, title=title, description=description)

    def _delete_position(self, doc_id, dictionary):
        keys_to_del = []
        for key in dictionary.keys():
            if doc_id in dictionary[key].keys():
                self.vocabulary[key] -= len(dictionary[key][doc_id])
                del dictionary[key][doc_id]
            if len(dictionary[key].keys()) == 0:
                keys_to_del.append(key)
        for kdl in keys_to_del:
            del dictionary[kdl]

    # todo: phase2?
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
            # bigram
            if not self.vocabulary[word]:
                del self.vocabulary[word]
            bis = proc_text.bigram_word(word)
            for bi in bis:
                self.bigram_indices[bi][word] -= 1
                if self.bigram_indices[bi][word] == 0:
                    del self.bigram_indices[bi][word]
                if len(self.bigram_indices[bi].keys()) == 0:
                    del self.bigram_indices[bi]

        self.collections_deleted.append(doc_id)
        print('Document', doc_id, 'deleted successfully')

    def stop_words(self, threshold=0.5):
        """lists stopwords based on overall frequency - threshold: frequency percentage"""
        word_freq = sum(self.vocabulary.values())
        stop_words = list(filter(lambda x: x[1] / word_freq >= threshold / 100., self.vocabulary.items()))
        print(', '.join(
            f'{x[0]}({x[1] / word_freq * 100.:.04f}%)' for x in sorted(stop_words, key=lambda x: x[1], reverse=True)))

    # part 2
    def posting_list_by_word(self, word: str, views: int = None):
        """retrieves posting list based on given word"""
        if views is not None and (self.talks_vectors is None or self.talks_vectors[-1] is None):
            self.classify()
        lang = self.lang
        if not self.dataset_loaded:
            self.prepare_text(word, lang)
            return
        term = proc_text.prepare_text(word, lang, verbose=False)[0]
        print_formatted_text(HTML(f'<skyblue>Term:</skyblue> <cyan>{term}</cyan>'))
        for idx, values in self.positional_indices_title.get(term, dict()).items():
            if views is not None and self._filtered(idx, views):
                continue
            print_match_doc(
                idx, positions_title=values, title=self.collections[idx][0], terms=[term],
                print_terms=False, lang=lang,
                description=None if idx not in self.positional_indices.get(term, dict()) else self.collections[idx][1],
                positions_description=self.positional_indices.get(term, dict()).get(idx, tuple())
            )
        for idx, values in self.positional_indices.get(term, dict()).items():
            if views is not None and self._filtered(idx, views):  # filtering by views
                continue
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
        print(*list(self.bigram_indices.get(bigram, dict()).keys()), sep=', ')

    def words_by_bigram_suggestion(self, args):
        if not args or not args[0]:
            return self.bigram_indices.keys()
        if len(args) > 1:
            return []
        return filter(lambda x: x.startswith(args[0]), self.bigram_indices.keys())

    def _load_vocabulary(self):
        self.vocabulary = dict()
        for term in self.positional_indices:
            self.vocabulary[term] = sum([len(value) for doc, value in self.positional_indices[term].items()])
        for term in self.positional_indices_title:
            if term not in self.vocabulary:
                self.vocabulary[term] = 0
            self.vocabulary[term] += sum([len(value) for doc, value in self.positional_indices_title[term].items()])

    def save(self):
        """saves indices without compression"""
        with open(self.positional_add, 'wb') as handle:
            pickle.dump(self.positional_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.positional_title_add, 'wb') as handle:
            pickle.dump(self.positional_indices_title, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.bigram_add, 'wb') as handle:
            pickle.dump(self.bigram_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("positional indices size:", os.stat(self.positional_add).st_size, ' bytes')
        print("positional titles indices size:", os.stat(self.positional_title_add).st_size, ' bytes')
        print("bigram indices size:", os.stat(self.bigram_add).st_size, ' bytes')

    def load(self):
        """loads uncompressed indices"""
        with open(self.positional_add, 'rb') as handle:
            self.positional_indices = pickle.load(handle)
        with open(self.positional_title_add, 'rb') as handle:
            self.positional_indices_title = pickle.load(handle)
        self._load_vocabulary()
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)

    # part 3
    def save_coded(self, coding: str = "gamma"):
        """compress and encode and save the indices - coding: ['gamma', 'variable']"""
        self._code_indices(coding)
        with open(self.coded_add, 'wb') as handle:
            pickle.dump(self.coded_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.bigram_add, 'wb') as handle:
            pickle.dump(self.bigram_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.coded_title_add, 'wb') as handle:
            pickle.dump(self.coded_title_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("coding:", coding)
        print("coded positional indices size:", os.stat(self.coded_add).st_size, 'bytes')
        print("coded positional titles indices size:", os.stat(self.coded_title_add).st_size, 'bytes')
        print("bigram indices size:", os.stat(self.bigram_add).st_size, 'bytes')

    def load_coded(self, coding: str = "gamma"):
        """load the saved coded indices - coding: ['gamma', 'variable']"""
        with open(self.coded_add, 'rb') as handle:
            self.coded_indices = pickle.load(handle)
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)
        with open(self.coded_title_add, 'rb') as handle:
            self.coded_title_indices = pickle.load(handle)
        self._decode_indices(coding)
        self._load_vocabulary()

    def load_coded_suggestion(self, args):
        codings = ['gamma', 'variable']
        if not args or not args[0]:
            return codings
        if len(args) > 1:
            return []
        return filter(lambda x: x.startswith(args[0]), codings)

    def save_coded_suggestion(self, args):
        return self.load_coded_suggestion(args)

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

    # part 4
    def fix_query(self, query: str):
        """fixes queries based on the available vocabulary and jaccard distance"""
        lang = self.lang
        dictionary = list(self.vocabulary.keys())
        fixed_query = []
        pre_query = proc_text.prepare_text(query, lang, False)
        for word in pre_query:
            if word in dictionary:
                fixed_query.append(word)
            else:
                fixed_query.append(wc.fix_word(word, dictionary))
        return ' '.join(fixed_query)

    def jaccard_dist(self, word1: str, word2: str):
        """calculates the jaccard distance of two words"""
        word_list1 = [word1[i:i + 2] for i in range(len(word1) - 1)]
        word_list2 = [word2[i:i + 2] for i in range(len(word2) - 1)]
        return wc.calc_jaccard(word_list1, word_list2)

    def edit_dist(self, word1: str, word2: str):
        """calculates the edit distance of two words"""
        return wc.calc_edit_distance(word1, word2)

    # part 5
    def _score_docs(self, query_terms, zone='title', docs: set = None):
        lang = self.lang
        dictionary = self.positional_indices if zone == 'description' else self.positional_indices_title
        collection_idx = 1 if zone == 'description' else 0
        query_scores = score_query(query_terms, dictionary, len(self.collections))
        result = dict()
        for term, term_score in zip(query_terms, query_scores):
            posting = dictionary.get(term, dict())
            if docs is not None:
                posting = {x: y for x, y in posting.items() if x in docs}
            for doc, positions in posting.items():
                if doc not in result:
                    result[doc] = 0
                result[doc] += logarithmic(len(positions)) * term_score

        for doc in result:
            result[doc] /= scale_lnc(self.collections[doc][collection_idx], lang)
        return result

    def sort_by_relevance(self, query: str, k: int = 10, views: int = None):
        """find top-k relevant documents based on search"""
        if views is not None and (self.talks_vectors is None or self.talks_vectors[-1] is None):
            self.classify()
        lang = self.lang
        self.prepare_text(query, lang)  # print the query terms
        query_terms = proc_text.prepare_text(query, lang, False)
        title_scores = self._score_docs(query_terms, 'title')
        description_scores = self._score_docs(query_terms, 'description')
        result = dict()
        for doc in description_scores:
            if views is not None and self._filtered(doc, views):  # filtering by views
                continue
            result[doc] = description_scores[doc] + title_scores.get(doc, 0)
        for doc_id, score in sorted(list(result.items())[:k], key=lambda x: x[1], reverse=True):
            print_match_doc(
                doc_id=doc_id, score=score, description=self.collections[doc_id][1], title=self.collections[doc_id][0],
                terms=query_terms, print_terms=False, lang=lang
            )

    def proximity_search(self, query: str, zone: str = 'title', window: int = 5, views: int = None):
        """performs a proximity search for query - zone: ['title','description'], window: proximity window-size"""
        lang = self.lang
        dictionary = self.positional_indices if zone == 'description' else self.positional_indices_title
        self.prepare_text(query, lang)  # print the query terms
        query_terms = proc_text.prepare_text(query, lang, False)
        if len(query_terms) < 2:
            print('query should at least contain two terms')
            return
        answer = set()
        for i in range(1, len(query_terms)):
            last_positions = dictionary.get(query_terms[i - 1], dict())
            cur_positions = dictionary.get(query_terms[i], dict())
            cur_answer = set()
            for doc in filter(lambda x: x in last_positions, cur_positions):
                # positional intersection
                for pos in cur_positions[doc]:
                    for last_pos in last_positions[doc]:
                        if last_pos < pos - window:
                            continue
                        if last_pos > pos + window:
                            break
                        if abs(pos - last_pos) <= window:
                            cur_answer.add(doc)
                            break
            if i == 1:
                answer = cur_answer
            answer = set(filter(lambda x: x in answer, cur_answer))
            if not answer:
                break
        if views is not None:
            answer = self._filter_resulting_talks(answer, views)
        result = self._score_docs(query_terms, zone, answer)
        for doc_id, score in sorted(list(result.items()), key=lambda x: x[1], reverse=True):
            print_match_doc(
                doc_id=doc_id, score=score,
                title=self.collections[doc_id][0] if zone == 'title' else None,
                description=self.collections[doc_id][1] if zone == 'description' else None,
                terms=query_terms, print_terms=False, lang=lang
            )

    # phase 2
    # part 1
    def _init_data(self, split='train'):
        data = pd.read_csv(f'files/{split}.csv')[['title', 'description', 'views']]
        data = pd.DataFrame(
            {'terms': (data['title'] + ' ' + data['description']).apply(proc_text.prepare_text),
             'views': data['views']})
        data['terms'] = data['terms'].apply(lambda x: {i: x.count(i) for i in x})
        vocab = proc_text.vocab(data['terms'])
        setattr(self, f'{split}_term_mapping', {x: y[0] for x, y in vocab.items()})
        if split == 'train':
            self.train_data = data.sample(frac=self.train_ratio, random_state=RANDOM_SEED)
            self.val_data = data.drop(self.train_data.index)
            self.train_vectors = ntn_vectorize(self.train_data, vocab)
            self.val_vectors = ntn_vectorize(self.val_data, vocab)
            self.train_term_mapping = {x: y[0] for x, y in vocab.items()}
        elif split == 'test':
            self.test_data = data
            self.test_vectors = ntn_vectorize(data, vocab)
        elif split == 'talks':
            self.talks_vectors = ntn_vectorize(data, vocab)

    def init_data(self):
        """initialize the train, val and test data splits for classification"""
        self._init_data('train')
        self._init_data('test')
        self._init_data('talks')

    def fit_models(self):
        """fit models on data"""
        self.models.append(classifiers.NaiveBayes())
        self.models.append(classifiers.KNN())
        for model in self.models:
            model.fit(*self.train_vectors, self.train_term_mapping)

    def fine_tune_models(self):
        """fine-tune models hyper parameters based on the validation split"""
        for model in self.models:
            print_formatted_text(HTML(f'<skyblue>Fine tuning:</skyblue> <cyan>{model}</cyan>'))
            model.fine_tune(*self.val_vectors, self.train_term_mapping)
            print_formatted_text(HTML(f'\tFine tuned: <bold>{model}</bold>'))
            break

    # part 2
    def classify(self):
        """classify the talks dataset with the best model"""
        if self.test_vectors is None:
            self.init_data()
        if not self.models:
            self.fit_models()
            self.fine_tune_models()
        if self.best_model is None:
            self.evaluate_models()
        self.talks_vectors = self.talks_vectors[0], self.talks_vectors[1], self.best_model.classify(
            self.talks_vectors[0], self.train_vectors[1], self.talks_term_mapping)

    def _filter_resulting_talks(self, results, views):
        if self.test_vectors is None or self.test_vectors[-1] is None:
            self.classify()
        docs = set(np.arange(len(self.talks_vectors[0]))[self.talks_vectors[-1] == views])
        return set(filter(lambda x: x in docs, results))

    def _filtered(self, doc_id, views):
        return self.talks_vectors[-1][doc_id] != views

    # part 3
    def evaluate_models(self):
        """evaluate the models and find the best one"""
        best_accuracy = 0.
        for model in self.models:
            test_res = model.evaluate(self.test_vectors[0], self.train_vectors[1], self.test_vectors[-1],
                                      self.test_term_mapping)
            if test_res['accuracy'] > best_accuracy:
                self.best_model = model
                best_accuracy = test_res['accuracy']
            val_res = model.evaluate(*self.val_vectors, self.train_term_mapping)
            train_res = model.evaluate(*self.train_vectors, self.train_term_mapping)
            result = mix_evaluation_results(train_results=train_res, val_results=val_res, test_results=test_res)
            print_evaluation_results(model, result)
            break
