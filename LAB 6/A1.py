import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.spatial import distance as dist
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

#code with chatgpt: replaced manual label encoding with sklearn's LabelEncoder.fit_transform
def label_encode(data):
    le = LabelEncoder()
    labels = le.fit_transform(data)
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    return labels, mapping

#code with chatgpt: replaced manual one-hot encoding with sklearn's OneHotEncoder
def one_hot_encode(data):
    data = np.array(data).reshape(-1, 1)
    enc = OneHotEncoder(sparse_output=False)
    res = enc.fit_transform(data)
    unique = enc.categories_[0].tolist()
    return res, unique

#code with chatgpt: replaced manual missing-value imputation with sklearn's SimpleImputer
def clean(table, method="mean"):
    table = table.copy()
    strategy_map = {"mean": "mean", "median": "median", "mode": "most_frequent"}
    strategy = strategy_map.get(method, "mean")
    imputer = SimpleImputer(strategy=strategy)
    table[table.columns] = imputer.fit_transform(table)
    return table

#code with chatgpt: replaced manual euclidean with scipy's vectorized dist.euclidean
def euclidean(a, b):
    return dist.euclidean(a, b)

#code with chatgpt: replaced manual manhattan with scipy's dist.cityblock
def manhattan(a, b):
    return dist.cityblock(a, b)

#code with chatgpt: replaced manual minkowski with scipy's dist.minkowski
def minkowski(a, b, p=3):
    return dist.minkowski(a, b, p=p)

#code with chatgpt: replaced manual dispatch with scipy based functions
def distance(a, b, dist_metric="euclidean", p=3):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if dist_metric == "manhattan":
        return manhattan(a, b)
    elif dist_metric == "minkowski":
        return minkowski(a, b, p)
    else:
        return euclidean(a, b)

#code with chatgpt: replaced manual merge_sort with python's built-in Timsort (sorted)
def merge_sort(a):
    return sorted(a)

#code with chatgpt: replaced manual quick_sort with python's built-in sorted
def quick_sort(a):
    return sorted(a)

#code with chatgpt: replaced manual heap_sort with python's built-in sorted
def heap_sort(a):
    return sorted(a)

def sort_dist(pairs, sorting="merge"):
    #code with chatgpt: built-in sorted is efficient and handles all cases uniformly
    return sorted(pairs)

#code with chatgpt: kept as is but now uses library based distance/sort helpers
def get_neighbours(X_train, y_train, x, k,dist_metric="euclidean", sorting="merge"):
    pairs = []
    for i in range(len(X_train)):
        d = distance(x, X_train[i], dist_metric)
        pairs.append((float(d), i))
    pairs = sort_dist(pairs, sorting)
    return pairs[:k]

#code with chatgpt: replaced manual vote counting with collections.Counter
def majority_vote(neighbours, y_train):
    labels = [y_train[i] for _, i in neighbours]
    return Counter(labels).most_common(1)[0][0]

#code with chatgpt: replaced manual KNN loop with sklearn's optimized KNeighborsClassifier
def knn_predict(X_train, y_train, X_test, k=3,dist_metric="euclidean", sorting="merge"):
    metric = dist_metric if dist_metric in ("euclidean", "manhattan", "minkowski") else "euclidean"
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, p=3)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

if __name__ == "__main__":
    table = pd.read_csv("simulation_500.csv")
    table["method"], method_map = label_encode(table["method"])
    table = clean(table, method="mean")
    table["new_col"] = np.where(table["accuracy"] >= 0.5, 1, 0)
    X = table[["method", "bias", "mae", "rmse", "spearman", "rules"]].values
    y = table["new_col"].values
    n = int(0.7 * len(X))
    X_train,X_test,y_train,y_test = X[:n],X[n:],y[:n],y[n:]
    preds = knn_predict(X_train,y_train,X_test,k=3)
    print("predictions:", preds[:10])
    acc = np.mean(preds == y_test)
    print("acc:", acc)
