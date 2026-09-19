import pandas as pd

def gini(y):
    p = y.value_counts(normalize=True)  #prob of each category
    return 1 - sum(p ** 2)              #gini = 1 - sum(p^2)

data = pd.read_csv("ACTG175.csv")       
print("Gini Index:", gini(data["treat"])) 
