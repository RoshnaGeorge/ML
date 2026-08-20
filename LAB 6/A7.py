import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#code with chatgpt: kept function, returns training data
def fit(X_train, y_train):
    return X_train, y_train #training data

#code with chatgpt: replaced manual KNN implementation with sklearn's optimized KNeighborsClassifier
def predict(X_train, y_train, X_test, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

#code with chatgpt: replaced manual accuracy computation with sklearn.metrics.accuracy_score
def score(y_test, pred):
    return accuracy_score(y_test, pred) #compare actual vs predicted values

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
