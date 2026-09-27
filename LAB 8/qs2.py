import matplotlib.pyplot as plt

def step(x):
    return 1 if x >= 0 else 0
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
        y = step(summ(a, b, w0, w1, w2))
        e = calc_err(t[i], y)
        sse += e**2
    return sse
def train(x, t, w0, w1, w2, lr):
    errs = []
    for ep in range(1, 1001):
        for i in range(len(x)):
            a, b = x[i]
            net = summ(a, b, w0, w1, w2)
            y = step(net)
            e = calc_err(t[i], y)
            w0, w1, w2 = upd(w0, w1, w2, a, b, e, lr)
        sse = sse_calc(x, t, w0, w1, w2)
        errs.append(sse)
        if sse <= 0.002:
            break
    return w0, w1, w2, ep, errs

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
t = [0, 0, 0, 1]

w0 = 10
w1 = 0.2
w2 = -0.75
lr = 0.05
w0, w1, w2, ep, errs = train(x, t, w0, w1, w2, lr)

print("no of epoch:", ep)
print("wts", w0, w1, w2)
print("AND gate:")

for i in range(len(x)):
    a, b = x[i]
    y = step(summ(a, b, w0, w1, w2))
    print(a, b, "->", y)

plt.plot(range(1, ep + 1), errs, marker='o')
plt.xlabel("epoch")
plt.ylabel("sse")
plt.title("epoch vs sse")
plt.grid()
plt.show()