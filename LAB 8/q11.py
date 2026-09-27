import numpy as np
from sklearn.neural_network import MLPClassifier

def train(x, t, hidden):
    model = MLPClassifier(hidden_layer_sizes=(hidden,), activation="tanh", solver="lbfgs", max_iter=2000, random_state=1)
    model.fit(x, t)
    return model

def predict(model, x):
    return model.predict(x)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_and = np.array([0, 0, 0, 1])
t_xor = np.array([0, 1, 1, 0])
model = train(x, t_and, 2)
y_and = predict(model, x)
print("AND Gate")
print("i/p ac pred")
for i in range(len(x)):
    print(x[i], "   ", t_and[i], "  ", y_and[i])
model = train(x, t_xor, 4)
y_xor = predict(model, x)
print("XOR Gate")
print("i/p ac pred")
for i in range(len(x)):
    print(x[i], "   ", t_xor[i], "  ", y_xor[i])
