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


def auto_cast(s):
    for fn in (dictify, boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s
