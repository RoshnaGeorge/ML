import numpy as np
from A4 import minkowski_distance
from A8 import mean


def init_centroids(data, k):
    return data[:k]


def assign(data, cents, order=2):
    cs = [[] for i in range(len(cents))]
    for p in data:
        ni = 0
        md = minkowski_distance(p, cents[0], order)
        for i in range(1, len(cents)):
            d = minkowski_distance(p, cents[i], order)
            if d < md:
                md = d
                ni = i
        cs[ni].append(p)
    return cs


def recompute(cs, oc):
    nc = []
    for i in range(len(cs)):
        c = cs[i]
        if len(c) == 0:
            nc.append(oc[i])
            continue
        ct = []
        for j in range(len(c[0])):
            v = []
            for p in c:
                v.append(p[j])
            ct.append(mean(v))
        nc.append(np.array(ct))
    return nc


def converged(old, new, tol=1e-6):
    for i in range(len(old)):
        if minkowski_distance(old[i], new[i], 2) > tol:
            return False
    return True


def kmeans(data, k, order=2, max_iter=100):
    cents = init_centroids(data, k)
    for it in range(max_iter):
        cs = assign(data, cents, order)
        nc = recompute(cs, cents)
        if converged(cents, nc):
            print("Converged in", it + 1, "iterations")
            return cs, nc
        cents = nc
    return cs, cents


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "..\\simulation_500.csv"
)

df = df.select_dtypes(include="number")
df = df.dropna()

data = df.to_numpy()

k = 3

cs, cents = kmeans(data, k)

print("\nFinal Centroids:\n")

for i in range(len(cents)):
    print(f"Centroid {i+1}")
    print(cents[i])
    print()

print("Cluster Sizes:\n")

for i in range(len(cs)):
    print(f"Cluster {i+1}: {len(cs[i])} points")

colors = ["red", "blue", "green", "orange", "purple"]

plt.figure(figsize=(8,6))

for i in range(len(cs)):
    c = np.array(cs[i])
    if len(c) > 0:
        plt.scatter(
            c[:,0],
            c[:,1],
            color=colors[i],
            label=f"Cluster {i+1}"
        )

cents = np.array(cents)

plt.scatter(
    cents[:,0],
    cents[:,1],
    color="black",
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("K-Means Clustering")
plt.legend()
plt.grid(True)
plt.show()
