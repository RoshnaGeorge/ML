import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

df = pd.read_csv("ACTG175.csv")
f = ["cd40", "age"]
X = df[f]
Y = df["treat"]
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X, Y)
plt.figure(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(dt,X,response_method="predict",alpha=0.5)
plt.scatter(X["cd40"],X["age"],c=Y,edgecolor="black",s=30)#actual data points
plt.xlabel("CD40")
plt.ylabel("Age")
plt.title("Decision Boundary using CD40 and Age")
plt.show()
