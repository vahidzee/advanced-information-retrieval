from .template import Classifier
import numpy as np
from prompt_toolkit.shortcuts import ProgressBar


class NaiveBayes(Classifier):
    def __init__(self, term_mapping=None):
        super().__init__(term_mapping)
        self.laplace_k = 0
        self.weights = None
        self.total_weights = None
        self.class_priors = None

    def fit(self, tf, idf, classes, terms_mapping):
        self.weights = np.stack([tf[classes == -1, :].sum(0), tf[classes == 1, :].sum(0)])
        self.total_weights = self.weights.sum(1)
        self.term_mapping = terms_mapping
        self.class_priors = np.log(np.array([len(classes[classes == -1]), len(classes[classes == 1])]) / len(classes))

    def classify(self, tf, idf, terms_mapping, **kwargs) -> np.array:
        # accounting for different vocabularies
        tf, idf = self._project_vectors(tf, idf, terms_mapping)
        # calculating results
        result = (tf * np.log((self.weights.reshape(2, 1, -1) + self.laplace_k) / (
                self.total_weights.reshape(-1, 1, 1) + self.laplace_k * len(self.term_mapping)))).sum(2)
        return np.argmax(result + self.class_priors.reshape(2, 1), axis=0) * 2 - 1

    def fine_tune(self, tf, idf, classes, terms_mapping):
        metric = []
        pb = ProgressBar()
        pb.__enter__()
        for k in pb(range(1, 30), label='Finding best smoothing k'):
            self.laplace_k = k
            metric.append((self.evaluate(tf, idf, classes, terms_mapping)[self.fine_tune_metric], k))
        pb.__exit__()
        self.laplace_k = max(metric)[1]

    def __repr__(self):
        return f'{self.__class__.__name__}-smoothing={self.laplace_k}'
