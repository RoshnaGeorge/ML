import math
import matplotlib.pyplot as plt

def step(x):
    if x >= 0:
        return 1
    else:
        return 0
def bipolar(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
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
t = [0, 1, 1, 0]
w0 = 10
w1 = 0.2
w2 = -0.75
lr = 0.05
ep_st, ws0, ws1, ws2, err_st = train(x, t, step,w0, w1, w2, lr)
tb = [-1, 1, 1, -1]
ep_bp, wb0, wb1, wb2, err_bp = train(x, tb, bipolar,w0, w1, w2, lr)
ep_sg, wg0, wg1, wg2, err_sg = train(x, t, sigmoid,w0, w1, w2, lr)
ep_rl, wr0, wr1, wr2, err_rl = train(x, t, relu,w0, w1, w2, lr)
print("XOR Gate")
print("epochs:")
print("step:", ep_st)
print("biploar :", ep_bp)
print("sigmoid:", ep_sg)
print("relu:", ep_rl)
print("sse:")
print("step:", err_st[-1])
print("bipolar:", err_bp[-1])
print("sigmoid:", err_sg[-1])
print("relu:", err_rl[-1])
plt.plot(range(1, len(err_st) + 1),err_st,label="step")
plt.plot(range(1, len(err_bp) + 1),err_bp,label="bipolar")
plt.plot(range(1, len(err_sg) + 1),err_sg,label="sigmoid")
plt.plot(range(1, len(err_rl) + 1),err_rl,label="relu")
plt.xlabel("epoch")
plt.ylabel("sse")
plt.title("xor:epooch vs sse")
plt.legend()
plt.grid()
plt.show()