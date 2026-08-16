import numpy as np
import pandas as pd


def euclidean_distance(point1, point2):
    return np.sqrt(
        np.sum(
            (np.array(point1) - np.array(point2)) ** 2
        )
    )


def weighted_knn_predict(training_data, training_labels, test_point, k):
    distances = []

    # Calculate distance from test point to every training point
    for i in range(len(training_data)):
        dist = euclidean_distance(
            test_point,
            training_data[i]
        )

        distances.append(
            (dist, training_labels[i])
        )

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Select K nearest neighbors
    k_nearest = distances[:k]

    # Store total weight for each class
    class_weights = {}

    for dist, label in k_nearest:

        # Weight = 1 / distance
        # Small value prevents division by zero
        weight = 1 / (dist + 1e-10)

        if label not in class_weights:
            class_weights[label] = 0

        class_weights[label] += weight

    # Class with highest total weight is the prediction
    prediction = max(
        class_weights,
        key=class_weights.get
    )

    return prediction


def main():

    # Load dataset
    df = pd.read_csv("simulation_500.csv")

    # Separate features and labels
    training_data = df.iloc[:, :-1].values
    training_labels = df.iloc[:, -1].values

    print("\nEnter values for the test point:")

    # Get test point from user
    test_point = []

    for column in df.columns[:-1]:
        value = float(input(f"Enter {column}: "))
        test_point.append(value)

    test_point = np.array(test_point)

    # Get K from user
    k = int(input("\nEnter the value of K: "))

    # Validate K
    if k <= 0 or k > len(training_data):
        print("Invalid value of K.")
        return

    # Predict using Weighted KNN
    prediction = weighted_knn_predict(
        training_data,
        training_labels,
        test_point,
        k
    )

    # Display result
    print("\nTest Point:", test_point)
    print("K:", k)
    print("Predicted Class:", prediction)


if __name__ == "__main__":
    main()