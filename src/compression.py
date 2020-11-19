def decode_gamma_code(bits: str):  # function to decode gamma code
    ind, cnt, gaps = 0, 0, list()
    while ind < len(bits):
        if bits[ind] == '1':
            ind += 1
            cnt += 1
        else:
            ind += 1
            num = '1' + bits[ind:ind + cnt]
            ind += cnt
            cnt = 0
            gaps.append(int(num, 2))
    indices = [gaps[0]]
    for i in range(len(gaps) - 1):
        indices.append(gaps[i + 1] + indices[i])
    return indices


def string_gamma_code(bits: str):  # function to produce gamma code of single number
    ans = '0' + bits[1:]
    ans = '1' * (len(bits) - 1) + ans
    return ans


def gamma_code(indices):  # function to produce gamma code from indices
    indices = list(sorted(indices))
    gaps = [indices[0]]
    for i in range(1, len(indices)):
        gaps.append(indices[i] - indices[i - 1])
    for i in range(len(gaps)):
        gaps[i] = "{0:b}".format(gaps[i])
    ans = ""
    for i in gaps:
        ans += string_gamma_code(i)
    return int(ans, 2)


def decode_variable_length(bits: str) -> list:  # function to return indices list from variable length bytes
    n = (len(bits) + 7) // 8
    bits = '0' * (8 * n - len(bits)) + bits
    num = ""
    gaps = []
    for i in range(n):
        temp_byte = bits[i * 8:i * 8 + 8]
        num += temp_byte[1:]
        if temp_byte[0] == '1':
            gaps.append(int(num, 2))
            num = ""
    indices = [gaps[0]]
    for i in range(len(gaps) - 1):
        indices.append(gaps[i + 1] + indices[i])
    return indices


def bits_to_variable_byte(bits: str):  # function to turn one bit string into the variable byte in string form
    n = len(bits) // 7
    m = len(bits) % 7
    ans = ""
    if len(bits) % 7 != 0:
        n += 1
    if n == 1:
        ans += '1'
        ans += '0' * (7 - len(bits))
        ans += bits
    else:
        ans += '0' * (8 - m)
        ans += bits[:m]
        for i in range(n - 2):
            ans += '0'
            ans += bits[m + (i * 7):m + (i * 7) + 7]
        ans += '1'
        ans += bits[len(bits) - 7:len(bits)]
    return ans


def variable_byte(indices):  # function to produce variable bytes from indices
    indices = list(sorted(indices))
    gaps = [indices[0]]
    for i in range(1, len(indices)):
        gaps.append(indices[i] - indices[i - 1])
    for i in range(len(gaps)):
        if gaps[i] < 0:
            print("aaaa", gaps[i], i, indices)
        gaps[i] = "{0:b}".format(gaps[i])
    ans = ""
    for i in gaps:
        ans += bits_to_variable_byte(i)
    return int(ans, 2)
