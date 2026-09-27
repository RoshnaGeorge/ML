import math
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward(a, b, v11, v12, v21, v22, w1, w2):
    h1 = sigmoid(a*v11 + b*v21)
    h2 = sigmoid(a*v12 + b*v22)
    o = sigmoid(h1*w1 + h2*w2)
    return h1, h2, o
def backprop(a, b, t, h1, h2, o,v11, v12, v21, v22, w1, w2, lr):
    eo = t - o
    d_o = eo * o * (1 - o)
    d_h1 = d_o * w1 * h1 * (1 - h1)
    d_h2 = d_o * w2 * h2 * (1 - h2)
    w1 = w1 + lr * d_o * h1
    w2 = w2 + lr * d_o * h2
    v11 = v11 + lr * d_h1 * a
    v21 = v21 + lr * d_h1 * b
    v12 = v12 + lr * d_h2 * a
    v22 = v22 + lr * d_h2 * b
    return v11, v12, v21, v22, w1, w2
def calc_sse(x, t, v11, v12, v21, v22, w1, w2):
    sse = 0
    for i in range(len(x)):
        a, b = x[i]
        h1, h2, o = forward(a, b,v11, v12, v21, v22,w1, w2)
        e = t[i] - o
        sse += e**2
    return sse
def train(x, t, v11, v12, v21, v22, w1, w2, lr):
    errs = []
    for ep in range(1, 1001):
        for i in range(len(x)):
            a, b = x[i]
            h1, h2, o = forward(a, b,v11, v12, v21, v22,w1, w2)
            v11, v12, v21, v22, w1, w2 = backprop(a, b, t[i],h1, h2, o,v11, v12, v21, v22,w1, w2,lr)
        sse = calc_sse(x, t,v11, v12, v21, v22,w1, w2)
        errs.append(sse)
        if sse <= 0.002:
            break
    return v11, v12, v21, v22, w1, w2, ep, errs
x = [[0, 0],[0, 1],[1, 0],[1, 1]]
t = [0, 0, 0, 1]
v11 = 0.5
v12 = -0.5
v21 = 0.5
v22 = -0.5
w1 = 0.5
w2 = 0.5
lr = 0.05
v11, v12, v21, v22, w1, w2, ep, errs = train(x, t,v11, v12, v21, v22,w1, w2,lr)
print("no of epochs:", ep)
print("final wts:")
print("v11 =", v11)
print("v12 =", v12)
print("v21 =", v21)
print("v22 =", v22)
print("w1  =", w1)
print("w2  =", w2)
print("AND:")
for i in range(len(x)):
    a, b = x[i]
    h1, h2, o = forward(a, b,v11, v12, v21, v22,w1, w2)
    y = 1 if o >= 0.5 else 0
    print(a, b,"->",round(o, 4),"->",y)
plt.plot(range(1, ep + 1),errs)
plt.xlabel("epoch")
plt.ylabel("sse")
plt.title("backprop-epoch vs sse")
plt.grid()
plt.show()