import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(fp):
    df = pd.read_csv(fp)
    return df

def prep_data(df):
    X = df.drop(columns=["arms", "Unnamed: 0"])
    y = df["arms"]
    return X, y

def split_data(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return Xtr, Xte, ytr, yte

def create_pipeline():
    pipe = Pipeline([("scaler", StandardScaler()),("model", SVC())])
    return pipe

def train_pipeline(pipe, Xtr, ytr):
    pipe.fit(Xtr, ytr)
    return pipe

def evaluate_pipeline(pipe, Xte, yte):
    yp = pipe.predict(Xte)
    acc = accuracy_score(yte, yp)
    pre = precision_score(yte, yp, average="weighted", zero_division=0)
    rec = recall_score(yte, yp, average="weighted", zero_division=0)
    f1 = f1_score(yte, yp, average="weighted", zero_division=0)
    return acc, pre, rec, f1

fp = "ACTG175_X_train.csv"
df = load_data(fp)
X, y = prep_data(df)
Xtr, Xte, ytr, yte = split_data(X, y)
pipe = create_pipeline()
pipe = train_pipeline(pipe, Xtr, ytr)
acc, pre, rec, f1 = evaluate_pipeline(pipe,Xte,yte)
print("pipeline")
print("acc:", acc)
print("precision:", pre)
print("recall:", rec)
print("f1:", f1)