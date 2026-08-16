import numpy as np
import pandas as pd

def label_encode(data):
    unique = []
    for value in data:
        if value not in unique:
            unique.append(value)
    labels = {}
    for i, value in enumerate(unique):
        labels[value] = i
    res = []
    for value in data:
        res.append(labels[value])
    return res, labels


def one_hot_encode(data):
    unique = []
    for value in data:
        if value not in unique:
            unique.append(value)
    res = []

    for value in data:
        row = []
        for item in unique:
            if value == item:
                row.append(1)
            else:
                row.append(0)
        res.append(row)
    return res, unique

def impute(table, method="mean"):
    table = table.copy()
    for column in table.columns:
        if table[column].isnull().any():
            if method == "mean":
                value = table[column].mean()
            elif method == "median":
                value = table[column].median()
            else:
                value = table[column].mode()[0]
            table[column] = table[column].fillna(value)
    return table

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def minkowski(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1.0 / p)

def distance(a, b, dist_metric="euclidean", p=3):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dist_metric == "euclidean":
        return euclidean(a, b)
    elif dist_metric == "manhattan":
        return manhattan(a, b)
    elif dist_metric == "minkowski":
        return minkowski(a, b, p)
    else:
        return euclidean(a, b)

def merge_sort(a):
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    result = []
    i = 0,j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result

def quick_sort(a):
    if len(a) <= 1:
        return a
    pivot = a[0]
    left = []
    right = []
    for i in range(1, len(a)):
        if a[i] < pivot:
            left.append(a[i])
        else:
            right.append(a[i])
    return quick_sort(left) + [pivot] + quick_sort(right)


def heapify(a, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and a[left] > a[largest]:
        largest = left
    if right < n and a[right] > a[largest]:
        largest = right
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        heapify(a, n, largest)

def heap_sort(a):
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        heapify(a, n, i)
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        heapify(a, i, 0)
    return a

def sort_dist(pairs, sorting="merge"):
    if sorting == "merge":
        return merge_sort(pairs)
    elif sorting == "quick":
        return quick_sort(pairs)
    elif sorting == "heap":
        return heap_sort(pairs)
    else:
        raise ValueError("error,must be one among the 3 only")

def get_neighbours(X_train, y_train, x, k,dist_metric="euclidean", sorting="merge"):
    pairs = []
    for i in range(len(X_train)):
        d = distance(x, X_train[i], dist_metric)
        pairs.append((float(d), i))
    pairs = sort_dist(pairs, sorting)
    return pairs[:k]

def majority_vote(neighbours, y_train):
    votes = {}
    for dist, i in neighbours:
        label = y_train[i]
        if label not in votes:
            votes[label] = 1
        else:
            votes[label] += 1
    best_label = None
    best_votes = 0
    for dist, i in neighbours:
        label = y_train[i]
        if votes[label] > best_votes:
            best_votes = votes[label]
            best_label = label
    return best_label

def knn_predict(X_train, y_train, X_test, k=3,dist_metric="euclidean", sorting="merge"):
    preds = []
    for x in X_test:
        neighbours = get_neighbours(X_train,y_train,x,k,dist_metric,sorting)
        preds.append(majority_vote(neighbours, y_train))
    return np.array(preds)

if __name__ == "__main__":
    table = pd.read_csv("simulation_500.csv")
    table["method"], method_map = label_encode(table["method"])
    table = impute(table, method="mean")
    table["new_col"] = np.where(table["accuracy"] >= 0.5, 1, 0)
    X = table[["method", "bias", "mae", "rmse", "spearman", "rules"]].values
    y = table["new_col"].values
    n = int(0.7 * len(X))
    X_train,X_test,y_train,y_test = X[:n],X[n:],y[:n],y[n:]
    preds = knn_predict(X_train,y_train,X_test,k=3)
    print("predictions:", preds[:10])
    acc = np.mean(preds == y_test)
    print("acc:", acc)