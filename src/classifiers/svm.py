from sklearn.svm import SVC
from .template import Classifier
from sklearn.decomposition import PCA
from prompt_toolkit.shortcuts import ProgressBar
import numpy as np

RANDOM_SEED = 666


class SVM(Classifier):
    def __init__(self, term_mapping=None, c=0.5, pca=0):
        super().__init__(term_mapping)
        self.clf = None
        self.train_tf = None
        self.train_idf = None
        self.train_classes = None
        self.c = c
        self.pca = PCA(n_components=pca) if pca else None

    def fit(self, tf=None, idf=None, classes=None, terms_mapping=None):
        tf = tf if tf is not None else self.train_tf
        idf = idf if idf is not None else self.train_idf
        classes = classes if classes is not None else self.train_classes
        if self.train_tf is None:
            self.train_tf, self.train_idf, self.train_classes, self.term_mapping = tf, idf, classes, terms_mapping
            if self.pca:
                self.pca.fit(tf * idf)
        train_points = tf * idf
        if self.pca:
            train_points = self.pca.transform(train_points)
        self.clf = SVC(C=self.c, random_state=RANDOM_SEED)
        self.clf.fit(train_points, classes)

    def classify(self, tf, idf, terms_mapping, **kwargs) -> np.array:
        # accounting for different vocabularies
        tf, idf = self._project_vectors(tf, idf, terms_mapping)
        vectors = tf * idf
        if self.pca:
            vectors = self.pca.transform(vectors)
        # calculating results
        return self.clf.predict(vectors)

    def fine_tune(self, tf, idf, classes, terms_mapping):
        metric = []
        pb = ProgressBar()
        pb.__enter__()
        for c in pb(np.linspace(0.5, 2, 4), label='Finding best C'):
            self.c = c
            self.fit()
            metric.append((self.evaluate(tf, idf, classes, terms_mapping, pb=pb)[self.fine_tune_metric], c))
        pb.__exit__()
        self.c = max(metric)[1]

    def __repr__(self):
        return f'{self.__class__.__name__}-C={self.c}'
