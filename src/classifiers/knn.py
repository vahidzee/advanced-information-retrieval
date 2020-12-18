from .template import Classifier
import numpy as np
from prompt_toolkit.shortcuts import ProgressBar


class KNN(Classifier):
    def __init__(self, term_mapping=None):
        super().__init__(term_mapping)
        self.k = 1
        self.cos_similarity = False
        self.low_memory = True
        self.train_data = None
        self.train_classes = None
        self.fine_tune_mode = False
        self.cache = None

    @staticmethod
    def _normalize_vectors(vector):
        return vector / np.sqrt((vector ** 2).sum(1, keepdims=True))

    def fit(self, tf, idf, classes, terms_mapping, pb=None):
        self.train_data = (tf * idf)
        if self.cos_similarity:
            self.train_data = self._normalize_vectors(self.train_data)
        if not self.low_memory:
            self.train_data = self.train_data.reshape(1, *self.train_data.shape)
        self.train_classes = classes
        self.term_mapping = terms_mapping

    def _scores_low_memory(self, vectors, pb=None):
        result = []
        for v in pb(vectors, label='Finding nearest neighbours') if pb is not None else vectors:
            if self.cos_similarity:
                result.append((v * self.train_data).sum(-1))
            else:
                result.append(np.square(v - self.train_data).sum(-1))
        return np.stack(result)

    def classify(self, tf, idf, terms_mapping, pb=None) -> np.array:
        # accounting for different vocabularies
        if self.fine_tune_mode is False or self.cache is None:
            tf, idf = self._project_vectors(tf, idf, terms_mapping)
            vectors = tf * idf
            if self.cos_similarity:
                vectors = self._normalize_vectors(vectors)
                scores = self._scores_low_memory(vectors, pb) if self.low_memory else \
                    (vectors.reshape(-1, 1, vectors.shape[-1]) * self.train_data).sum(-1)
            else:
                scores = self._scores_low_memory(vectors, pb) if self.low_memory else \
                    np.square(vectors.reshape(-1, 1, vectors.shape[-1]) - self.train_data).sum(-1)
        if self.fine_tune_mode and self.cache is None:
            self.cache = scores
        if self.fine_tune_mode:
            scores = self.cache
        args = np.argpartition(scores, -self.k)[:, -self.k:]  # Indices not sorted
        return np.where(self.train_classes[args[:, :self.k]].sum(-1) > 0, 1, -1)

    def fine_tune(self, tf, idf, classes, terms_mapping):
        metric = []
        self.fine_tune_mode = True
        with ProgressBar() as pb:
            for k in pb(range(2, 100), 'Finding best k'):
                self.k = k
                metric.append((self.evaluate(tf, idf, classes, terms_mapping, pb=pb)[self.fine_tune_metric], k))
        self.fine_tune_mode = False
        self.cache = None
        self.k = max(metric)[1]

    def __repr__(self):
        return f'{self.__class__.__name__}-k={self.k}'
