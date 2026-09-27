import math
import matplotlib.pyplot as plt
def bipolar(x):
    if x >= 0:
        return 1
    else:
        return -1
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def relu(x):
    if x > 0:
        return x
    else:
        return 0
def summ(a, b, w0, w1, w2):
    return w0 + w1*a + w2*b
def calc_err(t, y):
    return t - y
def upd(w0, w1, w2, a, b, e, lr):
    w0 = w0 + lr*e
    w1 = w1 + lr*e*a
    w2 = w2 + lr*e*b
    return w0, w1, w2
#sse
def sse_calc(x, t, w0, w1, w2, act):
    sse = 0
    for i in range(len(x)):
        a, b = x[i]
        net = summ(a, b, w0, w1, w2)
        y = act(net)
        e = calc_err(t[i], y)
        sse += e**2
    return sse
def train(x, t, act, w0, w1, w2, lr):
    errs = []
    for ep in range(1, 1001):
        for i in range(len(x)):
            a, b = x[i]
            net = summ(a, b, w0, w1, w2)
            y = act(net)
            e = calc_err(t[i], y)
            w0, w1, w2 = upd(w0, w1, w2,a, b, e, lr)
        sse = sse_calc(x, t,w0, w1, w2,act)
        errs.append(sse)
        if sse <= 0.002:
            break
    return ep, w0, w1, w2, errs
x = [[0, 0],[0, 1],[1, 0],[1, 1]]
t = [0, 0, 0, 1]
w0 = 10
w1 = 0.2
w2 = -0.75
lr = 0.05
tb = [-1, -1, -1, 1]
ep_b, wb0, wb1, wb2, err_b = train(x, tb, bipolar,w0, w1, w2, lr)
ep_s, ws0, ws1, ws2, err_s = train(x, t, sigmoid,w0, w1, w2, lr)
ep_r, wr0, wr1, wr2, err_r = train(x, t, relu,w0, w1, w2, lr)
print("iter:")
print("bipolar:", ep_b)
print("sigmoid:", ep_s)
print("relu:", ep_r)
print("final wts")
print("bipolar:")
print(wb0, wb1, wb2)
print("sigmoid:")
print(ws0, ws1, ws2)
print("relu:")
print(wr0, wr1, wr2)
plt.plot(range(1, ep_b + 1),err_b,label="bipolar")
plt.plot(range(1, ep_s + 1),err_s,label="sigmoid")
plt.plot(range(1, ep_r + 1),err_r,label="relu")
plt.xlabel("epochs")
plt.ylabel("sse")
plt.title("epoch vs sse")
plt.legend()
plt.grid()
plt.show()