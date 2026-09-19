import pandas as pd
import numpy as np

def bin_feat(x, bins=3):  #converting continuous feature into 3 categories
    edges = np.linspace(x.min(), x.max(), bins + 1)
    return pd.cut(x, edges)
def entropy(y):
    p = y.value_counts(normalize=True)
    return -sum(p * np.log2(p))
def info_g(y, x):
    gain = entropy(y)
    for value in x.unique():
        subset = y[x == value]
        gain -= (len(subset) / len(y)) * entropy(subset)
    return gain
def decision_tree(data, features, target): #best feature for the root node
    y = data[target]
    best_feature = None
    best_gain = -1
    for feature in features:
        x = data[feature]
        if x.nunique() > 5: #divide them into 3 groups if diff values
            x = bin_feat(x)
        gain = info_g(y, x)
        print(feature, ":", round(gain, 3))
        if gain > best_gain:
            best_gain = gain  #feature with highest gain
            best_feature = feature
    print("\nroot node:", best_feature)
    print("info gain:", round(best_gain, 3))

data = pd.read_csv("ACTG175.csv")
features = ["age", "wtkg", "hemo", "homo", "drugs", "karnof","oprior", "z30", "zprior", "preanti", "race", "gender","str2", "strat", "symptom", "cd40", "cd80"]
decision_tree(data, features, "treat")
