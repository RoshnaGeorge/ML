def label_encode(data):
    mp = {}
    enc = []

    for x in data:
        if x not in mp:
            mp[x] = len(mp)
        enc.append(mp[x])

    return enc, mp

def one_hot_encode(data):
    mp = {}

    for x in data:
        if x not in mp:
            mp[x] = len(mp)

    res = []

    for x in data:
        row = [0] * len(mp)
        row[mp[x]] = 1
        res.append(row)

    return res, mp