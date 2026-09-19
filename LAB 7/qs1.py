import pandas as pd             
import numpy as np               

def entropy(y):
    p = y.value_counts(normalize=True) #prob of each unique value in y
    return -sum(p * np.log2(p)) #H = -sum(p * log2(p))

data = pd.read_csv("ACTG175.csv")
print("Entropy:", entropy(data["treat"]))