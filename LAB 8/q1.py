def summ(a, b, w0, w1, w2):
    return w0 + w1*a + w2*b

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

def bipolar(x):
    if x >= 0:
        return 1
    else:
        return -1

def sigmoid(x):
    return 1 / (1 + 2.71828 ** (-x))

def tanh(x):
    return (2 / (1 + 2.71828 ** (-2*x))) - 1

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def leaky_relu(x):
    if x > 0:
        return x
    else:
        return 0.01 * x
    
def error(t, op):
    return t - op #target-o/p