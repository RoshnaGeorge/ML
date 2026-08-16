import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def fit(X_train, y_train):
    return X_train, y_train

def predict(X_train, y_train, X_test, k=3):
    pred = []
    for x in X_test:
        distances = []
        for i in range(len(X_train)):
            distance = np.sqrt(np.sum((X_train[i] - x) ** 2))
            distances.append((distance, i))
        distances.sort()
        neighbours = distances[:k]
        votes = {}
        for distance, i in neighbours:
            label = y_train[i]
            if label not in votes:
                votes[label] = 1
            else:
                votes[label] += 1
        best_label = None
        best_votes = 0
        for distance, i in neighbours:
            label = y_train[i]
            if votes[label] > best_votes:
                best_votes = votes[label]
                best_label = label
        pred.append(best_label)
    return np.array(pred)

def score(y_test, pred):
    return np.mean(y_test == pred)

table = pd.read_csv("simulation_500.csv")
table["new_col"] = np.where(table["accuracy"] >= 0.5, 1, 0)
X = table[["method", "bias", "mae", "rmse", "spearman", "rules"]].values
y = table["new_col"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Fit
X_train, y_train = fit(X_train, y_train)
# Predict
pred = predict(X_train, y_train, X_test, k=3)
# Accuracy
acc = score(y_test, pred)
print("acc:", acc)
print("first 20 PREDICTIONS:", pred[:20])
print("first 20 ACTUALS:", y_test[:20])
print("No of crct predictions:", np.sum(pred == y_test))