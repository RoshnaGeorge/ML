import pandas as pd
import numpy as np

def entropy(y):
    p = y.value_counts() / len(y)     
    return -sum(p * np.log2(p))        
def info_g(y, x):
    total = entropy(y)                  #entropy before split
    for v in x.unique():               
        part = y[x == v]                
        total -= len(part) / len(y) * entropy(part)  #subtracting the weighted entropy
    return total                        

data = pd.read_csv("ACTG175.csv")     
features = ["age", "wtkg", "hemo", "homo", "drugs", "karnof","oprior", "z30", "zprior", "preanti", "race", "gender","str2", "strat", "symptom", "cd40", "cd80"]
y = data["treat"]                      
best_feature = ""                     
best_gain = -1                         #initially
for col in features:
    x = data[col]                      
    if x.nunique() > 5:#converting the numerical features into 3 categorical bins
        x = pd.cut(x, 3)
    gain = info_g(y, x)     
    print(col, ":", round(gain, 3))    
    if gain > best_gain:
        best_gain = gain
        best_feature = col #best feature so far
print("\nroot node:", best_feature)     
