import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
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


def create_models():
    models = {
        "SVM": Pipeline([("scaler", StandardScaler()),("model", SVC())]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),
        "CatBoost": CatBoostClassifier(iterations=100,verbose=False,random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100,random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100,eval_metric="mlogloss",random_state=42),
        "Naive Bayes": GaussianNB()
    }
    return models

def train_models(models, Xtr, ytr):
    for name, model in models.items():
        model.fit(Xtr, ytr)
    return models

def get_metrics(model, Xte, yte):
    yp = model.predict(Xte)
    acc = accuracy_score(yte, yp)
    pre = precision_score(yte, yp, average="weighted", zero_division=0)
    rec = recall_score(yte, yp, average="weighted", zero_division=0)
    f1 = f1_score(yte, yp, average="weighted", zero_division=0)
    return acc, pre, rec, f1

def compare_models(models, Xte, yte):
    results = []
    for name, model in models.items():
        acc, pre, rec, f1 = get_metrics(model, Xte, yte)
        results.append([name,acc,pre,rec,f1])
    res = pd.DataFrame(results,columns=["classifier","acc","precision","recall","f1"])
    return res


fp = "ACTG175_X_train.csv"
df = load_data(fp)
X, y = prep_data(df)
Xtr, Xte, ytr, yte = split_data(X, y)
models = create_models()
models = train_models(models, Xtr, ytr)
res = compare_models(models, Xte, yte)
print("classifier perf comparison:")
print(res.to_string(index=False))