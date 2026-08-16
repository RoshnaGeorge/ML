import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("simulation_500.csv")
df["class"] = np.where(df["accuracy"] >= 0.5, 1, 0)
X = df[["method", "bias", "mae", "rmse", "spearman", "rules"]].values
y = df["class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#fit
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
#predict
pred = neigh.predict(X_test)
print("PREDICTIONS:", pred[:20])
print("ACTUALS:", y_test[:20])
print("No of crct predictions:", np.sum(pred == y_test))
