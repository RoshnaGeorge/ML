import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("ACTG175.csv")
f = ["age", "wtkg", "hemo", "homo", "drugs", "karnof","oprior", "z30", "zprior", "preanti", "race", "gender","str2", "strat", "symptom", "cd40", "cd80"]
X = df[f]
Y = df["treat"]
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=1)
dt = DecisionTreeClassifier(random_state=1)
dt.fit(Xtr, Ytr)
Ypr = dt.predict(Xte)
acc = accuracy_score(Yte, Ypr)
print("Accuracy:", acc)
plt.figure(figsize=(20, 10))
plot_tree(dt,feature_names=f,class_names=["0", "1"],filled=True,rounded=True)
plt.savefig("ACTG175_tree.png")
plt.show()
print("Decision tree saved as ACTG175_tree.png")
