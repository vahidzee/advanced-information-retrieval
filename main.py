from __future__ import unicode_literals
from hazm import *
import pandas as pd
import xml.etree.ElementTree as ET
import collections
import nltk
import string
from nltk.stem import PorterStemmer
import pickle
from sys import getsizeof


class MIR:
    def __init__(self):
        self.persian_wikis = []
        self.ted_talk_title = []
        self.ted_talk_desc = []
        self.positional_indices = dict()  # key: word value: dict(): key: doc ID, value: list of positions
        self.coded_indices = dict()  # key: word value: dict(): key: doc ID, value: bytes of indices
        self.bigram_indices = dict()  # key: bi-gram value: dict(): key: word, value: collection freq
        self.collections = []
        self.collections_deleted = []  # vector indicating whether the corresponding document is deleted or not

        self.positional_add = 'src/pos.pickle'
        self.bigram_add = 'src/bi.pickle'
        self.coded_add = 'src/coded.pickle'

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
        tokens = self.prepare_text(document, lang, False)

        # Bi-gram
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

    def posting_list_by_word(self, word, lang):
        token = self.prepare_text(word, lang, verbose=False)[0]
        print(self.positional_indices[token])

    def words_by_bigram(self, bigram):
        print(self.bigram_indices[bigram].keys())

    def save_indices(self):
        with open(self.positional_add, 'wb') as handle:
            pickle.dump(self.positional_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.bigram_add, 'wb') as handle:
            pickle.dump(self.bigram_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_coded_indices(self):
        self.code_indices()
        with open(self.coded_add, 'wb') as handle:
            pickle.dump(self.coded_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.bigram_add, 'wb') as handle:
            pickle.dump(self.bigram_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_indices(self):
        with open(self.positional_add, 'rb') as handle:
            self.positional_indices = pickle.load(handle)
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)

    def load_coded_indices(self):
        with open(self.coded_add, 'rb') as handle:
            self.coded_indices = pickle.load(handle)
        with open(self.bigram_add, 'rb') as handle:
            self.bigram_indices = pickle.load(handle)
        self.decode_indices()

    def code_indices(self, coding="s"):
        for word in self.positional_indices:
            self.coded_indices[word] = dict()
            for doc in self.positional_indices[word]:
                self.coded_indices[word][doc] = self.gamma_code(
                    self.positional_indices[word][doc]) if coding == "gamma" else self.variable_byte(
                    self.positional_indices[word][doc])

    def decode_indices(self, coding="s"):
        for word in self.coded_indices:
            self.positional_indices[word] = dict()
            for doc in self.coded_indices[word]:
                self.positional_indices[word][doc] = self.decode_gamma_code(format(
                    self.coded_indices[word][doc], "b")) if coding == "gamma" else self.decode_variable_length(
                    format(self.coded_indices[word][doc], "b"))

    @staticmethod
    def prepare_text(text, lang, verbose=True):
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

    @staticmethod
    def bits_to_variable_byte(bits: str):  # function to turn one bit string into the variable byte in string form
        n = len(bits) // 7
        m = len(bits) % 7
        ans = ""
        if len(bits) % 7 != 0:
            n += 1
        if n == 1:
            ans += '1'
            ans += '0' * (7 - len(bits))
            ans += bits
        else:
            ans += '0' * (8 - m)
            ans += bits[:m]
            for i in range(n - 2):
                ans += '0'
                ans += bits[m + (i * 7):m + (i * 7) + 7]
            ans += '1'
            ans += bits[len(bits) - 7:len(bits)]
        return ans

    def variable_byte(self, indices):  # function to produce variable bytes from indices
        indices = list(sorted(indices))
        gaps = [indices[0]]
        for i in range(1, len(indices)):
            gaps.append(indices[i] - indices[i - 1])
        for i in range(len(gaps)):
            if gaps[i] < 0:
                print("aaaa", gaps[i], i, indices)
            gaps[i] = "{0:b}".format(gaps[i])
        ans = ""
        for i in gaps:
            ans += self.bits_to_variable_byte(i)
        return int(ans, 2)

    @staticmethod
    def decode_variable_length(bits: str) -> list:  # function to return indices list from variable length bytes
        n = (len(bits) + 7) // 8
        bits = '0' * (8 * n - len(bits)) + bits
        num = ""
        gaps = []
        for i in range(n):
            temp_byte = bits[i * 8:i * 8 + 8]
            num += temp_byte[1:]
            if temp_byte[0] == '1':
                gaps.append(int(num, 2))
                num = ""
        indices = [gaps[0]]
        for i in range(len(gaps) - 1):
            indices.append(gaps[i + 1] + indices[i])
        return indices

    @staticmethod
    def string_gamma_code(bits: str):  # function to produce gamma code of single number
        ans = '0' + bits[1:]
        ans = '1' * (len(bits) - 1) + ans
        return ans

    def gamma_code(self, indices):  # function to produce gamma code from indices
        indices = list(sorted(indices))
        gaps = [indices[0]]
        for i in range(1, len(indices)):
            gaps.append(indices[i] - indices[i - 1])
        for i in range(len(gaps)):
            gaps[i] = "{0:b}".format(gaps[i])
        ans = ""
        for i in gaps:
            ans += self.string_gamma_code(i)
        return int(ans, 2)

    @staticmethod
    def decode_gamma_code(bits: str):  # function to decode gamma code
        ind = 0
        cnt = 0
        gaps = []
        while ind < len(bits):
            if bits[ind] == '1':
                ind += 1
                cnt += 1
            else:
                ind += 1
                num = '1' + bits[ind:ind + cnt]
                ind += cnt
                cnt = 0
                gaps.append(int(num, 2))
        indices = [gaps[0]]
        for i in range(len(gaps) - 1):
            indices.append(gaps[i + 1] + indices[i])
        return indices

    @staticmethod
    def calc_jaccard(A: list, B: list) -> float:  # calculates the jaccard distance of two sets
        same_cnt = 0
        for i in A:
            if i in B:
                same_cnt += 1
        return same_cnt / (len(A) + len(B) - same_cnt)

    def fix_query(self, query: str, lang: str):  # fixes queries considering their languages
        dictionary = list(self.positional_indices.keys())
        fixed_query = []
        pre_query = self.prepare_text(query, lang, False)
        for word in pre_query:
            if word in dictionary:
                fixed_query.append(word)
            else:
                fixed_query.append(self.fix_word(word, dictionary))
        return ' '.join(fixed_query)

    def get_jaccard_list(self, word: str,
                         dictionary) -> list:  # returns 10 closest words to a word according to the jaccard distance
        word_list = [word[i:i + 2] for i in range(len(word) - 1)]
        j_dists = {}
        for w in dictionary:
            temp_list = [w[i:i + 2] for i in range(len(w) - 1)]
            j_dists[w] = self.calc_jaccard(word_list, temp_list)
        return [list(dict(sorted(j_dists.items(), key=lambda x: x[1], reverse=True)).keys())[:10],
                list(dict(sorted(j_dists.items(), key=lambda x: x[1], reverse=True)).values())[:10]]

    @staticmethod
    def calc_edit_distance(A: str, B: str) -> int:
        n = len(A)
        m = len(B)
        c = dict()
        c[(0, 0)] = 0
        for i in range(m):
            c[(0, i + 1)] = i + 1
        for i in range(n):
            c[(i + 1, 0)] = i + 1
        for j in range(1, m + 1):
            for i in range(1, n + 1):
                if A[i - 1] == B[j - 1]:
                    c[(i, j)] = c[(i - 1, j - 1)]
                else:
                    c[(i, j)] = 1 + min([c[(i, j - 1)], c[(i - 1, j)], c[(i - 1, j - 1)]])
        return c[(n, m)]

    def fix_word(self, word: str, dictionary: list) -> str:
        jaccard_closest = self.get_jaccard_list(word, dictionary)
        min_ed = 100
        max_jd = 0
        chosen_word = ''
        for i in range(len(jaccard_closest[0])):
            w = jaccard_closest[0][i]
            w_ed = self.calc_edit_distance(word, w)
            if w_ed < min_ed:
                chosen_word = w
                min_ed = w_ed
                max_jd = jaccard_closest[1][i]
            elif w_ed == min_ed and max_jd < jaccard_closest[1][i]:
                chosen_word = w
                min_ed = w_ed
                max_jd = jaccard_closest[1][i]
        return chosen_word


mir = MIR()
mir.load_datasets('talks')
