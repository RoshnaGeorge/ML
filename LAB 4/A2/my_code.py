import numpy as np


def label_encode(data):
    unq = [] #take all unique values
    for x in data:
        if x not in unq:
            unq.append(x)
    mp = {}
    for i, v in enumerate(unq): 
        mp[v] = i
    enc = [] 
    for x in data:
        enc.append(mp[x])# appended the values
    return enc, mp


def one_hot_encode(data):
    mp = {}  # code with chatgpt

    for x in data:  # code with chatgpt
        if x not in mp:  # code with chatgpt
            mp[x] = len(mp)  # code with chatgpt

    res = []
    for x in data:
        row = [0] * len(mp)  # code with chatgpt
        row[mp[x]] = 1  # code with chatgpt
        res.append(row)

    return res, mp  # code with chatgpt
# basically just use dictionary/table


def minkowski_distance(a, b, p):
    return np.sum(np.abs(a - b) ** p) ** (1 / p)


def dot(A, B):
    r = 0
    for i in range(len(A)):
        r += A[i] * B[i]
    return r


def norm(V):
    s = 0
    for i in range(len(V)):
        s += V[i] ** 2
    return s ** 0.5


def mean(data):
    t = 0
    for v in data:
        t += v
    return t / len(data)


def variance(data):
    m = mean(data)
    t = 0
    for v in data:
        t += (v - m) ** 2
    return t / len(data)


def std(data):
    return variance(data) ** 0.5