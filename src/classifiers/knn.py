import collections

from .template import Classifier
import numpy as np
from prompt_toolkit.shortcuts import ProgressBar
from sklearn.decomposition import PCA


class KNN(Classifier):
    def __init__(self, k=1, cos_similarity=True, low_memory=True, pca=0, term_mapping=None):
        super().__init__(term_mapping)
        self.k = k
        self.cos_similarity = cos_similarity
        self.low_memory = low_memory
        self.train_data = None
        self.train_classes = None
        self.fine_tune_mode = False
        self.pca = PCA(n_components=pca) if pca else None
        self.cache = None

    @staticmethod
    def _normalize_vectors(vector):
        return vector / np.sqrt((vector ** 2).sum(1, keepdims=True))

    def fit(self, tf, idf, classes, terms_mapping, pb=None):
        self.train_data = (tf * idf)
        if self.pca:
            self.pca.fit(self.train_data)
            self.train_data = self.pca.transform(self.train_data)
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
                result.append(np.stack([(v * self.train_data[i]).sum(-1) for i in range(self.train_data.shape[0])]))
            else:
                result.append(
                    np.stack([np.square(v - self.train_data[i]).sum(-1) for i in range(self.train_data.shape[0])]))
        return np.stack(result)

    def classify(self, tf, idf, terms_mapping, pb=None) -> np.array:
        # accounting for different vocabularies
        if self.fine_tune_mode is False or self.cache is None:
            tf, idf = self._project_vectors(tf, idf, terms_mapping)
            vectors = tf * idf
            if self.pca:
                vectors = self.pca.transform(vectors)
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
        pb = ProgressBar()
        pb.__enter__()
        evaluation = collections.OrderedDict()
        for k in pb([1, 5, 9], 'Finding best k'):
            self.k = k
            results = self.evaluate(tf, idf, classes, terms_mapping, pb=pb)
            evaluation[k] = results
            metric.append((results[self.fine_tune_metric], k))
        pb.__exit__()
        self.fine_tune_mode = False
        self.cache = None
        self.k = max(metric)[1]
        return evaluation

    def __repr__(self):
        return f'{self.__class__.__name__}-k={self.k}'
