import numpy as np

def minkowski_distance(a, b, p):
    return np.linalg.norm(a - b, ord=p) #coded using chatgpt

if __name__ == "__main__":
    a = np.array([7, 3, 4])
    b = np.array([17, 6, 9])
    p = int(input("Enter order p: "))
    d = minkowski_distance(a, b, p)
    print("minkowski dist:", d)
    if p == 1:
        print("Manhattan.")
    elif p == 2:
        print("Euclidean.")