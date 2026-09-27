import pandas as pd
import matplotlib.pyplot as plt
def summ(a, b, w0, w1, w2):
    return w0 + w1*a + w2*b
def step(x):
    if x >= 0:
        return 1
    else:
        return 0
def calc_err(t, y):
    return t - y
def upd(w0, w1, w2, a, b, e, lr):
    w0 = w0 + lr*e
    w1 = w1 + lr*e*a
    w2 = w2 + lr*e*b
    return w0, w1, w2
def sse_calc(x, t, w01, w11, w21, w02, w12, w22):
    sse = 0
    for i in range(len(x)):
        a = x.iloc[i, 0]
        b = x.iloc[i, 1]
        if t.iloc[i] == 0:
            t1 = 1
            t2 = 0
        else:
            t1 = 0
            t2 = 1
        y1 = step(summ(a, b, w01, w11, w21))
        y2 = step(summ(a, b, w02, w12, w22))
        e1 = calc_err(t1, y1)
        e2 = calc_err(t2, y2)
        sse += e1**2 + e2**2
    return sse
def train(x, t, w01, w11, w21, w02, w12, w22, lr):
    errs = []
    for ep in range(1, 1001):
        for i in range(len(x)):
            a = x.iloc[i, 0]
            b = x.iloc[i, 1]
            if t.iloc[i] == 0:
                t1 = 1
                t2 = 0
            else:
                t1 = 0
                t2 = 1
            y1 = step(summ(a, b, w01, w11, w21))
            y2 = step(summ(a, b, w02, w12, w22))
            e1 = calc_err(t1, y1)
            e2 = calc_err(t2, y2)
            w01, w11, w21 = upd(w01, w11, w21, a, b, e1, lr)
            w02, w12, w22 = upd(w02, w12, w22, a, b, e2, lr)
        sse = sse_calc(x, t, w01, w11, w21, w02, w12, w22)
        errs.append(sse)
        print("epoch:", ep, "err", sse)
        if sse <= 0.002:
            break
    return w01, w11, w21, w02, w12, w22, ep, errs
data = pd.read_csv("ACTG175.csv")
x = data[["age", "wtkg"]]
t = data["treat"]
w01 = 10
w11 = 0.2
w21 = -0.75
w02 = 10
w12 = 0.2
w22 = -0.75
lr = 0.05
w01, w11, w21, w02, w12, w22, ep, errs = train(x, t, w01, w11, w21, w02, w12, w22, lr)
print("final wts:")
print("op1:")
print("W01 =", w01)
print("W11 =", w11)
print("W21 =", w21)
print("op2:")
print("W02 =", w02)
print("W12 =", w12)
print("W22 =", w22)
print("epochs need:", ep)
plt.plot(range(1, len(errs) + 1), errs)
plt.xlabel("epoch")
plt.ylabel("sse")
plt.title("epoch vs err")
plt.show()
