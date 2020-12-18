from sklearn.ensemble import RandomForestClassifier
from .template import Classifier
from prompt_toolkit.shortcuts import ProgressBar
import numpy as np
import collections

RANDOM_SEED = 666


class RandomForest(Classifier):
    def __init__(self, term_mapping=None):
        super().__init__(term_mapping)
        self.clf = None
        self.train_tf = None
        self.train_idf = None
        self.train_classes = None
        self.num_estimators = 100
        self.criterion = 'gini'
        self.max_depth = 5

    def fit(self, tf=None, idf=None, classes=None, terms_mapping=None):
        tf = tf if tf is not None else self.train_tf
        idf = idf if idf is not None else self.train_idf
        classes = classes if classes is not None else self.train_classes
        if self.train_tf is None:
            self.train_tf, self.train_idf, self.train_classes, self.term_mapping = tf, idf, classes, terms_mapping
        self.clf = RandomForestClassifier(n_estimators=self.num_estimators, max_depth=self.max_depth,
                                          random_state=RANDOM_SEED, criterion=self.criterion)
        self.clf.fit(tf * idf, classes)

    def classify(self, tf, idf, terms_mapping, **kwargs) -> np.array:
        # accounting for different vocabularies
        tf, idf = self._project_vectors(tf, idf, terms_mapping)
        vectors = tf * idf
        # calculating results
        return self.clf.predict(vectors)

    def fine_tune(self, tf, idf, classes, terms_mapping):
        metric = []
        pb = ProgressBar()
        evaluation = collections.OrderedDict()
        pb.__enter__()
        for k in pb(range(5, 110, 20), label='Finding best depth'):
            self.max_depth = k
            self.fit()
            results = self.evaluate(tf, idf, classes, terms_mapping)
            evaluation[k] = results
            metric.append((results[self.fine_tune_metric], k))
        pb.__exit__()
        self.max_depth = max(metric)[1]
        return evaluation

    def __repr__(self):
        return f'{self.__class__.__name__}-max-depth={self.max_depth}'
