import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_data(fp):
    df = pd.read_csv(fp)
    return df

def prep_data(df):
    X = df.drop(columns=["arms", "Unnamed: 0"])
    y = df["arms"]
    return X, y

def split_data(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
    return Xtr, Xte, ytr, yte

def tune_prc(Xtr, ytr):
    pipe = Pipeline([("sc", StandardScaler()),("prc", Perceptron(random_state=42))])
    prm = {
        "prc__penalty": [None, "l2", "l1", "elasticnet"],
        "prc__alpha": np.logspace(-6, -2, 10),
        "prc__max_iter": [500, 1000, 1500, 2000],
        "prc__tol": [1e-2, 1e-3, 1e-4],
        "prc__eta0": [0.01, 0.1, 1.0, 10.0]
    }
    rs = RandomizedSearchCV(pipe,prm,n_iter=20,cv=5,scoring="accuracy",random_state=42,n_jobs=-1)
    rs.fit(Xtr, ytr)
    return rs

def tune_mlp(Xtr, ytr):
    pipe = Pipeline([("sc", StandardScaler()),("mlp", MLPClassifier(random_state=42))])
    prm = {"mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        "mlp__activation": ["identity", "tanh", "relu"],
        "mlp__solver": ["adam", "sgd"],
        "mlp__alpha": np.logspace(-5, -2, 10),
        "mlp__learning_rate_init": [0.0001, 0.001, 0.01],
        "mlp__batch_size": [32, 64, 128],
        "mlp__max_iter": [300, 500, 700]
    }

    rs = RandomizedSearchCV(pipe,prm,n_iter=20,cv=5,scoring="accuracy",random_state=42,n_jobs=-1)
    rs.fit(Xtr, ytr)
    return rs

def eval_model(m, Xte, yte):
    yp = m.predict(Xte)
    acc = accuracy_score(yte, yp)
    return acc

fp = "ACTG175_X_train.csv"
df = load_data(fp)
X, y = prep_data(df)
Xtr, Xte, ytr, yte = split_data(X, y)
prs = tune_prc(Xtr, ytr)
mls = tune_mlp(Xtr, ytr)
prc_acc = eval_model(prs.best_estimator_, Xte, yte)
mlp_acc = eval_model(mls.best_estimator_, Xte, yte)
print("perceptron best params:")
print(prs.best_params_)
print("\nperceptron cv acc:")
print(prs.best_score_)
print("\nperceptron test acc:")
print(prc_acc)
print("\nmlp best params:")
print(mls.best_params_)
print("\nmlp cv acc:")
print(mls.best_score_)
print("\nmlp test acc:")
print(mlp_acc)