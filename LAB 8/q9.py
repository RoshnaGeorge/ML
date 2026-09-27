import numpy as np
def summ(a, b, w0, w1, w2):
    return w0 + w1*a + w2*b
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def forward(a, b, w1, w2, w3, w4, w5, w6, b1, b2, b3):
    #hidden layer
    h1 = sigmoid(summ(a, b, b1, w1, w2))
    h2 = sigmoid(summ(a, b, b2, w3, w4))
    # this is the o/p layer
    o = sigmoid(summ(h1, h2, b3, w5, w6))
    return h1, h2, o
def backprop(a, b, t, h1, h2, o, w1, w2, w3, w4, w5, w6, b1, b2, b3, lr):
    eo = t - o  #o/p layer error
    d_o = eo * o * (1 - o) #o/p delta
    #hidden layer deltas
    d_h1 = h1 * (1 - h1) * w5 * d_o
    d_h2 = h2 * (1 - h2) * w6 * d_o
    #updates the o/p weights
    w5 = w5 + lr * d_o * h1
    w6 = w6 + lr * d_o * h2
    b3 = b3 + lr * d_o
    #updates all the hidden weights
    w1 = w1 + lr * d_h1 * a
    w2 = w2 + lr * d_h1 * b
    b1 = b1 + lr * d_h1
    w3 = w3 + lr * d_h2 * a
    w4 = w4 + lr * d_h2 * b
    b2 = b2 + lr * d_h2
    return w1, w2, w3, w4, w5, w6, b1, b2, b3
def train(x, t, w1, w2, w3, w4, w5, w6, b1, b2, b3, lr):
    errs = []
    for ep in range(1, 10001):
        sse = 0
        for i in range(len(x)):
            a, b = x[i]
            #this is the forward propagation
            h1, h2, o = forward(a, b, w1, w2, w3, w4, w5, w6, b1, b2, b3)
            e = t[i] - o
            sse = sse + e ** 2
            #for backpropagation
            w1, w2, w3, w4, w5, w6, b1, b2, b3 = backprop(a, b, t[i], h1, h2, o, w1, w2, w3, w4, w5, w6, b1, b2, b3, lr)
        errs.append(sse)
        #convergence
        if sse <= 0.002:
            break
    return w1, w2, w3, w4, w5, w6, b1, b2, b3, ep, errs
#XOR gate data
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
t = [0, 1, 1, 0]
#initial weights
w1 = 0.01
w2 = 0.02
w3 = 0.03
w4 = 0.04
w5 = 0.05
w6 = -0.01
#initial bias
b1 = 0.01
b2 = 0.01
b3 = 0.01
lr = 0.05
#train
w1, w2, w3, w4, w5, w6, b1, b2, b3, ep, errs = train(x, t, w1, w2, w3, w4, w5, w6, b1, b2, b3, lr)
print("epochs:", ep)
print("err:", errs[-1])
print("final wts:")
print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("w4 =", w4)
print("w5 =", w5)
print("w6 =", w6)
