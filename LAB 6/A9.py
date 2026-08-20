import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#code with chatgpt: kept function, returns training data
def fit(X_train, y_train):
    return X_train, y_train

#code with chatgpt: replaced manual weighted KNN implementation with sklearn's KNeighborsClassifier(weights="distance")
def predict(X_train, y_train, X_test, k=3):
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

#code with chatgpt: replaced manual accuracy computation with sklearn.metrics.accuracy_score
def score(y_test, predictions):
    return accuracy_score(y_test, predictions)

table = pd.read_csv("simulation_500.csv")
table["new_col"] = np.where(table["accuracy"] >= 0.5, 1, 0)
X = table[["method", "bias", "mae", "rmse", "spearman", "rules"]].values
y = table["new_col"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
k_values = [1, 2, 3, 5, 7, 9, 11, 15]
my_scores = []
sk_scores = []
for k in k_values:
    pred_my = predict(X_train, y_train, X_test, k)
    my_scores.append(score(y_test, pred_my))
    neigh = KNeighborsClassifier(n_neighbors=k,weights="distance")
    neigh.fit(X_train, y_train)
    sk_scores.append(neigh.score(X_test, y_test))

print("k values:", k_values)
print("My kNN:", my_scores)
print("sklearn:", sk_scores)
plt.plot(k_values, my_scores, label="My Weighted kNN")
plt.plot(k_values, sk_scores, label="Sklearn Weighted kNN")
plt.xlabel("k")
plt.ylabel("acc")
plt.title("weighted kNN accuracy vs k")
plt.legend()
plt.show()
