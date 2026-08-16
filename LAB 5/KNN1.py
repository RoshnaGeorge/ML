import numpy as np
import pandas as pd
from collections import Counter


def euclidean_distance(point1, point2):
    return np.sqrt(
        np.sum(
            (np.array(point1) - np.array(point2)) ** 2
        )
    )


def knn_predict(training_data, training_labels, test_point, k):
    distances = []

    for i in range(len(training_data)):
        dist = euclidean_distance(
            test_point,
            training_data[i]
        )

        distances.append(
            (dist, training_labels[i])
        )

    distances.sort(key=lambda x: x[0])

    k_nearest_labels = [
        label for _, label in distances[:k]
    ]

    prediction = Counter(
        k_nearest_labels
    ).most_common(1)[0][0]

    return prediction


def main():
    df = pd.read_csv("simulation_500.csv")
    training_data = df.iloc[:, :-1].values
    training_labels = df.iloc[:, -1].values
    print("\nEnter values for the test point:")
    test_point = []
    for column in df.columns[:-1]:
        value = float(input(f"Enter {column}: "))
        test_point.append(value)
    test_point = np.array(test_point)
    k = int(input("\nEnter the value of K: "))
    if k <= 0 or k > len(training_data):
        print("Invalid value of K.")
        return
    prediction = knn_predict(
        training_data,
        training_labels,
        test_point,
        k
    )
    print("\nTest Point:", test_point)
    print("K:", k)
    print("Predicted Class:", prediction)

if __name__ == "__main__":
    main()
