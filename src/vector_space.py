import math
from .text_processing import prepare_text

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
    return [score / scale for score in scores]


def score_query(query_terms: list, dictionary: dict, n: int):
    dfs = [len(dictionary.get(term, dict())) for term in query_terms]
    tfs = [query_terms.count(term) for term in query_terms]
    return ltc(tfs, dfs, n)
