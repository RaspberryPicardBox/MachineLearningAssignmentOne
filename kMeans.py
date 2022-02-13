import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def compute_euclidean_distance(vec_1, vec_2):
    distance = np.linalg.norm(vec_1 - vec_2)
    return distance


def initialise_centroids(dataset, k=2):
    centroids = []
    shuffleData = deepcopy(dataset)
    np.random.shuffle(shuffleData)  # Shuffle the data temporarily
    for i in range(k):
        centroids.append(list(shuffleData[i]))  # And then choose the k first random elements
    return centroids


def errorPlotter(errors, k):
    # Plot the error value against the iteration step
    plt.title("Error per Iteration of K-Means k={}".format(k))
    plt.plot(errors.keys(), errors.values())
    plt.xlabel("Iteration Step")
    plt.ylabel("Objective Function Value")
    plt.show()


def kmeans(dataset, k=2):
    centroids = initialise_centroids(dataset, k)  # Get list of original random centroids
    centroidsOld = np.zeros(np.shape(centroids))  # Create a new "old" list
    clusters = np.zeros(len(dataset))

    errors = {}  # Keep track of iteration step and error

    error = compute_euclidean_distance(np.array(centroids), np.array(centroidsOld))  # Compute the original error
    errors[0] = error

    runs = 0
    while error != 0 or runs <= 300:  # Limit at 300 iterations
        runs += 1
        for i in range(len(dataset)):  # For every value in the dataset
            distances = np.linalg.norm(dataset[i]-centroids, axis=1)  # Calculate the distance between the values and each centroid
            cluster = np.argmin(distances)  # Find the minimum distance to a centroid and cluster the value
            clusters[i] = cluster
        centroidsOld = deepcopy(centroids)  # Copy the previous centroids before updating
        for i in range(k):
            points = [dataset[j] for j in range(len(dataset)) if clusters[j] == i]  # Find all the values of certain k
            centroids[i] = np.mean(points, axis=0)  # Calculate the mean of the values and set as new centroids
        error = compute_euclidean_distance(np.array(centroids), np.array(centroidsOld))  # Compute the new error between the new centroids and the old
        if error == 0:
            break
        errors[runs] = error

    centroids = np.array(centroids)
    cluster_assigned = np.zeros((len(dataset), 5))
    for i in range(len(cluster_assigned)):
        cluster_assigned[i] = np.append(dataset[i], clusters[i])  # Add the cluster value to the values

    errorPlotter(errors, k)

    return centroids, cluster_assigned
