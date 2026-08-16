#select the right k value(Selecting the optimal value of K)
#calculate the euclidean distance of that test point to all the other data points.(Calculating distance)
#sort the data points and select the top k data points.(Finding Nearest Neighbors)
#Voting for Classification or Taking Average for Regression
#https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/

import numpy as np
from collections import Counter #Counter is used to count the occurrences of elements in a list or iterable. In KNN after finding the k nearest neighbor labels Counter helps count how many times each label appears.

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

#Training Data, Labels and Test Point
