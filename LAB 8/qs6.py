import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def normalize(x):
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    return (x - mn) / (mx - mn)
def summ(x, w, b):
    return np.dot(x, w) + b
def calc_err(t, y):
    return t - y
def upd(w, b, x, e, y, lr):
    d = e * y * (1 - y)
    w = w + lr * d * x
    b = b + lr * d
    return w, b
def calc_sse(x, t, w, b):
    sse = 0
    for i in range(len(x)):
        net = summ(x[i], w, b)
        y = sigmoid(net)
        e = calc_err(t[i], y)
        sse += e ** 2
    return sse
def train(x, t, w, b, lr, max_ep=10000):
    errs = []
    for ep in range(1, max_ep + 1):
        for i in range(len(x)):
            net = summ(x[i], w, b)
            y = sigmoid(net)
            e = calc_err(t[i], y)
            w, b = upd(w, b,x[i],e, y,lr)
        sse = calc_sse(x, t, w, b)
        errs.append(sse)
        if sse <= 0.002:
            break
    return w, b, ep, errs
def predict(x, w, b):
    y = sigmoid(summ(x, w, b))
    if y >= 0.5:
        return 1
    else:
        return 0
x = np.array([[20, 6, 2, 386],[16, 3, 6, 289],[27, 6, 2, 393],[19, 1, 2, 110],[24, 4, 2, 280],[22, 1, 5, 167],[15, 4, 2, 271],[18, 4, 2, 274],[21, 1, 4, 148],[16, 2, 4, 198]], dtype=float)
t = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
x = normalize(x)
w = np.zeros(4)
b = 0
lr = 10
w, b, ep, errs = train(x, t, w, b, lr)
print("no of epochs:", ep)
print("final wts:")
print(w)
print("final bias:")
print(b)
print("cust classifier:")
print("cust actual predict")
for i in range(len(x)):
    y = predict(x[i], w, b)
    actual = "High" if t[i] == 1 else "Low"
    predicted = "High" if y == 1 else "Low"
    print("C_" + str(i + 1),"     ",actual,"    ",predicted)