import numpy as np

def dot(A, B):#code with chatgpt
    return sum(x * y for x, y in zip(A, B))

def norm(V):#code with chatgpt
    return sum(x ** 2 for x in V) ** 0.5

A = [3, 4, 1]
B = [2, 0, 5]
d1 = dot(A, B)
n1 = norm(A)
n2 = norm(B)
d2 = np.dot(A, B)
n3 = np.linalg.norm(A)
n4 = np.linalg.norm(B)
print(d1,d2)
print(n1,n2)