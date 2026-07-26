import numpy as np

def minkowski_distance(a, b, p):
     return np.sum(np.abs(a - b) ** p) ** (1 / p)


if __name__ == "__main__":
    a = np.array([7, 3, 4])
    b = np.array([17, 6, 9])

    p = int(input("Enter order (p): "))

    d = minkowski_distance(a, b, p)

    print("Minkowski Distance =", d)

    if p == 1:
        print("This is Manhattan Distance.")

    elif p == 2:
        print("This is Euclidean Distance.")
