from .text_processing import prepare_text


def calc_jaccard(a: list, b: list) -> float:  # calculates the jaccard distance of two sets
    same_cnt = sum([1 for i in a if i in b])
    return same_cnt / (len(a) + len(b) - same_cnt)


def calc_edit_distance(a: str, b: str) -> int:
    n, m, c = len(a), len(b), {(0, 0): 0}
    for i in range(m):
        c[(0, i + 1)] = i + 1
    for i in range(n):
        c[(i + 1, 0)] = i + 1
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                c[(i, j)] = c[(i - 1, j - 1)]
            else:
                c[(i, j)] = 1 + min([c[(i, j - 1)], c[(i - 1, j)], c[(i - 1, j - 1)]])
    return c[(n, m)]


def get_jaccard_list(word: str, dictionary: list) -> list:
    """returns 10 closest words to a word according to the jaccard distance"""
    word_list = [word[i:i + 2] for i in range(len(word) - 1)]
    j_dists = {}
    for w in dictionary:
        temp_list = [w[i:i + 2] for i in range(len(w) - 1)]
        j_dists[w] = calc_jaccard(word_list, temp_list)
    return [list(dict(sorted(j_dists.items(), key=lambda x: x[1], reverse=True)).keys())[:10],
            list(dict(sorted(j_dists.items(), key=lambda x: x[1], reverse=True)).values())[:10]]


def fix_word(word: str, dictionary: list) -> str:
    jaccard_closest = get_jaccard_list(word, dictionary)
    min_ed = 100
    max_jd = 0
    chosen_word = ''
    for i in range(len(jaccard_closest[0])):
        w = jaccard_closest[0][i]
        w_ed = calc_edit_distance(word, w)
        if w_ed < min_ed:
            chosen_word = w
            min_ed = w_ed
            max_jd = jaccard_closest[1][i]
        elif w_ed == min_ed and max_jd < jaccard_closest[1][i]:
            chosen_word = w
            min_ed = w_ed
            max_jd = jaccard_closest[1][i]
    return chosen_word


def fix_query(self, query: str, lang: str):  # fixes queries considering their languages
    dictionary = list(self.positional_indices.keys())
    fixed_query = []
    pre_query = c(query, lang, False)
    for word in pre_query:
        if word in dictionary:
            fixed_query.append(word)
        else:
            fixed_query.append(fix_word(word, dictionary))
    return ' '.join(fixed_query)
