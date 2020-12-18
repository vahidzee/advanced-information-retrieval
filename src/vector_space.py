import math
from .text_processing import prepare_text
import numpy as np

log = math.log10


def logarithmic(x):
    return log(1 + x)


def scale_lnc(text, lang):
    terms = prepare_text(text, lang, False)
    tfs = [logarithmic(terms.count(term)) for term in terms]
    return math.sqrt(sum(score ** 2 for score in tfs))


def ltc(tfs, dfs, n):
    # l: log(1 + count(term, terms)
    # t: log(n/df)
    # c: sqrt(sum(square(scores)))
    ltfs = [logarithmic(tf) for tf in tfs]
    tdfs = [log(n / df) if df else 0 for df in dfs]
    scores = [x * y for x, y in zip(ltfs, tdfs)]
    scale = math.sqrt(sum(score ** 2 for score in scores))
    return [score / scale if scale else 0 for score in scores]


def score_query(query_terms: list, dictionary: dict, n: int):
    dfs = [len(dictionary.get(term, dict())) for term in query_terms]
    tfs = [query_terms.count(term) for term in query_terms]
    return ltc(tfs, dfs, n)


def ntn_vectorize(data, vocab_dict):
    data_terms = data['terms']
    tf = np.zeros((len(data_terms), len(vocab_dict)))
    idf = np.zeros((len(vocab_dict),))
    for (idx, df) in vocab_dict.values():
        idf[idx] = df
    idf = np.log(idf)
    for i, terms in enumerate(data_terms):
        for term in terms:
            tf[i][vocab_dict[term][0]] = terms[term]
    return tf, idf, data['views'].to_numpy() if 'views' in data else None
