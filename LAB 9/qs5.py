import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lime.lime_tabular import LimeTabularExplainer

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
    pipe = Pipeline([("scaler", StandardScaler()),("model", SVC(probability=True))])
    return pipe

def train_pipeline(pipe, Xtr, ytr):
    pipe.fit(Xtr, ytr)
    return pipe

def create_explainer(Xtr, ytr):
    explainer = LimeTabularExplainer(Xtr.values,feature_names=Xtr.columns.tolist(),class_names=sorted(ytr.unique().astype(str)),mode="classification",random_state=42)
    return explainer

def explain_prediction(explainer, pipe, Xte, index):
    instance = Xte.iloc[index].values
    exp = explainer.explain_instance(instance,pipe.predict_proba,num_features=10)
    pred = pipe.predict(Xte.iloc[[index]])[0]
    return pred, exp

fp = "ACTG175_X_train.csv"
df = load_data(fp)
X, y = prep_data(df)
Xtr, Xte, ytr, yte = split_data(X, y)
pipe = create_pipeline()
pipe = train_pipeline(pipe, Xtr, ytr)
explainer = create_explainer(Xtr, ytr)
pred, exp = explain_prediction(explainer,pipe,Xte,0)
print("pred Class:", pred)
print("\nLIME explaination:")
for feature, weight in exp.as_list():
    print(feature, ":", weight)