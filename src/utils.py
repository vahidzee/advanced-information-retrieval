from prompt_toolkit import print_formatted_text, HTML
from .text_processing import prepare_text


def boolify(s):
    if s == 'True' or s == 'true' or s == 'yes' or s == 'Yes':
        return True
    if s == 'False' or s == 'false' or s == 'no' or s == 'No':
        return False
    raise ValueError("cast error")


def dictify(s: str):
    if ':' not in s:
        raise ValueError("cast error")
    else:
        res = dict()
        pairs = s.split(';')
        for pair in pairs:
            key, val = pair.split(':')
            res[key] = auto_cast(val)
        return res


def nonify(s: str):
    if s.lower() == 'none':
        return None
    else:
        raise ValueError("cast error")


def auto_cast(s):
    for fn in (dictify, boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s


def highlight(text, terms_set, lang):
    ftext = []
    for word in text.split(' '):
        terms = prepare_text(word, lang)
        if terms and terms[0] in terms_set:
            ftext.append(f'<u>{word}</u>')
        else:
            ftext.append(word)
    return ' '.join(ftext)


def print_match_doc(doc_id: int, score: float = None, title: str = None, description: str = None,
                    positions_title: list = tuple(),
                    positions_description: list = tuple(), terms: list = None, print_terms=False,
                    lang='eng'
                    ):
    if terms and print_terms:
        sign = 'terms' if len(terms) > 1 else 'term'
        print_formatted_text(HTML(f'<skyblue>{sign}:</skyblue> <cyan>{" ".join(terms)}</cyan>'))

    terms_set = set() if terms is None else set(terms)
    fscore = f' - <skyblue>Score:</skyblue> <cyan>{score:.04f}</cyan>' if score is not None else ''
    print_formatted_text(HTML(f'<skyblue>Doc ID:</skyblue> <cyan>{doc_id}</cyan>{fscore}'))

    if positions_title:
        print_formatted_text(
            HTML(f'\t<skyblue>Title Positions:</skyblue> <cyan>{", ".join(str(i) for i in positions_title)}</cyan>'))

    ftitle = highlight(title, terms_set, lang) if title and terms_set else title
    if ftitle:
        print_formatted_text(
            HTML(f'\t<skyblue>Title:</skyblue> {ftitle}'))
    if positions_description:
        print_formatted_text(HTML(f'\t<skyblue>Description Positions:</skyblue> \
<cyan>{", ".join(str(i) for i in positions_description)}</cyan>'))
    fdescription = highlight(description, terms_set, lang) if description and terms_set else description
    if fdescription:
        print_formatted_text(
            HTML(f'\t<skyblue>Description:</skyblue>\n\t{fdescription}'))
