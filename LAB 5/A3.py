import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("simulation_500.csv")
df["class"] = np.where(df["accuracy"] >= 0.5, 1, 0)
X = df[["method", "bias", "mae", "rmse", "spearman", "rules"]].values
y = df["class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)