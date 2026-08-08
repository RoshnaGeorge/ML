import numpy as np


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


def minkowski_distance(a, b, p):
    return np.linalg.norm(a - b, ord=p) #coded using chatgpt


def dot(A, B): #code with chatgpt
    return sum(x * y for x, y in zip(A, B))


def norm(V): #code with chatgpt
    return sum(x ** 2 for x in V) ** 0.5


def mean(data): #code with chatgpt
    return sum(data) / len(data)


def variance(data): #code with chatgpt
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def std(data): #code with chatgpt
    return variance(data) ** 0.5