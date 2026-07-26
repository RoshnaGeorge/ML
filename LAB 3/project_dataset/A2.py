def label_encode(data):
    unq = []
    for x in data:
        if x not in unq:
            unq.append(x)
    mp = {}
    for i, v in enumerate(unq):
        mp[v] = i
    enc = []
    for x in data:
        enc.append(mp[x])
    return enc, mp


def one_hot_encode(data):
    unq = []
    for x in data:
        if x not in unq:
            unq.append(x)
    res = []
    for x in data:
        r = []
        for u in unq:
            if x == u:
                r.append(1)
            else:
                r.append(0)
        res.append(r)
    return res, unq
