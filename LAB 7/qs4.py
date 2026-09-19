import pandas as pd
import numpy as np

data = pd.read_csv("ACTG175.csv")
def bin_feature(x, bins=3, method="width"):
    if method == "width":
        edges = np.linspace(x.min(), x.max(), bins + 1) #range into equal-width bins
        return pd.cut(x, edges)
    elif method == "freq":
        return pd.qcut(x, bins) #data is divided so each bin has roughly the same number of values

print("default params:")
print(bin_feature(data["cd40"]).value_counts().sort_index())
print("\nwidth:")
print(bin_feature(data["cd40"], 3, "width").value_counts().sort_index())
print("\nfrequency binning:")
print(bin_feature(data["cd40"], 4, "freq").value_counts().sort_index())
