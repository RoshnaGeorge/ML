import matplotlib.pyplot as plt
def step(x):
    if x >= 0:
        return 1
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
def sse_calc(x, t, w0, w1, w2):
    sse = 0
    for i in range(len(x)):
        a, b = x[i]
        net = summ(a, b, w0, w1, w2)
        y = step(net)
        e = calc_err(t[i], y)
        sse += e**2
    return sse
def train(x, t, w0, w1, w2, lr):
    for ep in range(1, 1001):
        for i in range(len(x)):
            a, b = x[i]
            net = summ(a, b, w0, w1, w2)
            y = step(net)
            e = calc_err(t[i], y)
            w0, w1, w2 = upd(w0, w1, w2,a, b, e, lr)
        sse = sse_calc(x, t,w0, w1, w2)
        if sse <= 0.002:
            break
    return ep
x = [[0, 0],[0, 1],[1, 0],[1, 1]]
t = [0, 0, 0, 1]
w0 = 10
w1 = 0.2
w2 = -0.75
lrs = [0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1.0]
epochs = []
for lr in lrs:
    ep = train(x, t,w0, w1, w2,lr)
    epochs.append(ep)
print("lr       epoch")
for i in range(len(lrs)):
    print(lrs[i], "          ", epochs[i])
plt.plot(lrs, epochs, marker='o')
plt.xlabel("lr")
plt.ylabel("no. of epochs")
plt.title("lr vs no. of epochs")
plt.grid()
plt.show()