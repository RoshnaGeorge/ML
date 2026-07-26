import numpy as np


def dot(A, B):
    r = 0
    for i in range(len(A)):
        r += A[i] * B[i]
    return r

def norm(V):
    s = 0
    for i in range(len(V)):
        s += V[i] ** 2
    return s ** 0.5


A = [3, 4, 1]
B = [2, 0, 5]

d1 = dot(A, B)
n1 = norm(A)
n2 = norm(B)

d2 = np.dot(A, B)
n3 = np.linalg.norm(A)
n4 = np.linalg.norm(B)

print("Dot Product Comparison:")
print(f"  Custom : {d1}")
print(f"  NumPy  : {d2}")
print(f"  Match? : {d1 == d2}")

print("\nEuclidean Norm of A:")
print(f"  Custom : {n1}")
print(f"  NumPy  : {n3}")
print(f"  Match? : {abs(n1 - n3) < 0.000001}")

print("\nEuclidean Norm of B:")
print(f"  Custom : {n2}")
print(f"  NumPy  : {n4}")
print(f"  Match? : {abs(n2 - n4) < 0.000001}")
