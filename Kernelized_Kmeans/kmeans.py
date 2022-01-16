import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt

def kmeans_lloyds_algo(X, k, max_iterations):
    # Function to implement LLyod's algorithm (K means)

    # Random initialization of centroids
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    points = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)

    # Estimating centre coordinates.
    # Assign data to its closest center
    # Find the optimal centers
    iterations = 1
    while iterations<=max_iterations:
        centroids = np.vstack([X[points == i, :].mean(axis=0) for i in range(k)])
        temp_cluster = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)
        if np.array_equal(points, temp_cluster):
            break
        points = temp_cluster
        iterations += 1

    # Plot the estimated centres and the centroid coordinated
    plt.scatter(X[:, 0], X[:, 1], c=points, s=50, cmap='viridis')
    plt.show()
    return points

if __name__ == '__main__':

    # Three types of data where LLoyd's algorithm fails
    filePath1 = "data/concentric_circles.txt"
    filePath2 = "data/moons.txt"
    filePath3 = 'data/uneven_distributed_data.txt'

    X1,k1 = np.loadtxt(filePath1, delimiter=" "), 2
    X2,k2 = np.loadtxt(filePath2, delimiter=" "), 2
    X3,k3 = np.loadtxt(filePath3, delimiter=" "), 3

    X,k = X1,k1
    kmeans_lloyds_algo(X, k, 1000)




