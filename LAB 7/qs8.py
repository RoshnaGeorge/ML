import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("ACTG175.csv")
f = ["age", "wtkg", "hemo", "homo", "drugs", "karnof","oprior", "z30", "zprior", "preanti", "race", "gender","str2", "strat", "symptom", "cd40", "cd80"]
X = df[f]
Y = df["treat"]
dt = DecisionTreeClassifier(random_state=1)
#test values
params = {"criterion": ["gini", "entropy"],"max_depth": [3, 5, 7, 10, None],"min_samples_split": [2, 5, 10],"min_samples_leaf": [1, 2, 4]}
# 5-fold cross-validation
grid = GridSearchCV(dt,params,cv=5,scoring="accuracy")
grid.fit(X, Y)
print("best hyperparams:")
print(grid.best_params_)
print("best acc:", grid.best_score_)
