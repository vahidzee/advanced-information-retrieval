import abc
import numpy as np


class Classifier(abc.ABC):
    def __init__(self, term_mapping=None):
        # initialize the necessary parameters
        self.term_mapping = term_mapping
        self.fine_tune_metric = 'accuracy'

    def fit(self, tf, idf, classes, terms_mapping):
        raise NotImplementedError

    def fine_tune(self, tf, idf, classes, terms_mapping):
        raise NotImplementedError

    def classify(self, tf, idf, terms_mapping) -> np.array:
        """:return np.array of predicted classes"""
        raise NotImplementedError

    def _project_vectors(self, tf, idf, terms_mapping, project_idf=False):
        tf_vector = np.zeros((tf.shape[0], len(self.term_mapping)))
        idf_vector = np.zeros((len(self.term_mapping),))
        trans, idxs = [], []
        for term in terms_mapping:
            if term in self.term_mapping:
                idxs.append(self.term_mapping[term])
                trans.append(terms_mapping[term])
        tf_vector[:, idxs] = tf[:, trans]
        # idf_vector[idxs] = idf[trans]
        return tf_vector, idf_vector if project_idf else idf

    def evaluate(self, tf, idf, classes, test_terms_mapping, **kwargs):
        predicted_classes = self.classify(tf, idf, test_terms_mapping, **kwargs)
        accuracy = (predicted_classes == classes).sum() / len(classes)
        pos_precision = (predicted_classes[classes == 1] == 1).sum() / (predicted_classes == 1).sum()
        neg_precision = (predicted_classes[classes == -1] == -1).sum() / (predicted_classes == -1).sum()
        pos_recall = (predicted_classes[classes == 1] == 1).sum() / (classes == 1).sum()
        neg_recall = (predicted_classes[classes == -1] == -1).sum() / (classes == -1).sum()
        pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
        neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall)
        return {
            'accuracy': accuracy,
            'pos': {'precision': pos_precision, 'recall': pos_recall, 'f1': pos_f1},
            'neg': {'precision': neg_precision, 'recall': neg_recall, 'f1': neg_f1}, }

    def __repr__(self):
        return f'{self.__class__.__name__}'
