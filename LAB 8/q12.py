import pandas as pd
from sklearn.neural_network import MLPClassifier
def train(x, t, hidden):
    model = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", solver="lbfgs", max_iter=2000, random_state=1)
    model.fit(x, t)
    return model
def predict(model, x):
    return model.predict(x)
data = pd.read_csv("ACTG175.csv")
x = data[["age", "wtkg"]]
t = data["treat"]
model = train(x, t, 4)
y = predict(model, x)
print("ac:")
print(t.values)
print("pred:")
print(y)
accuracy = model.score(x, t)
print("acc:", accuracy * 100, "%")
