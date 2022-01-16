# Implementation of kernelized version of k-means algorithm

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel


def kernel_distance(x_1, x_2):
    # Implementation of different kernels

    if kernel == 'rbf_kernel':
        return rbf_kernel(x_1, x_2, var)

    if kernel == 'linear_kernel':
        return linear_kernel(x_1, x_2)

    if kernel == 'quadratic_kernel':
        return polynomial_kernel(x_1, x_2, 2, var)


def get_term_1(i, data, clusters):
    # Calculate the first term in derived equation
    def calculate_kernel_dist(dataI, cluster_points):
        x_1 = dataI.reshape(1, dataI.shape[0])
        result = 0
        for i in range(0, cluster_points.shape[0]):
            x_2 = cluster_points[i, :].reshape(1, cluster_points[i, :].shape[0])
            result = result + kernel_distance(x_1, x_2)
        result = 2 * result / cluster_points.shape[0]
        return result

    term_1 = np.ndarray(shape=(0, 1))
    for j in range(0, data.shape[0]):
        dist = calculate_kernel_dist(data[j, :], np.array(clusters[i]))
        term_1 = np.concatenate((term_1, dist), axis=0)
    return (np.array(term_1))


def get_term_2(i, data, clusters):
    # Calculate the second term in derived equation
    def calculate_kernel_dist(cluster_points):
        result = 0
        for i in range(0, cluster_points.shape[0]):
            x_1 = cluster_points[i, :].reshape(1, cluster_points[i, :].shape[0])
            for j in range(0, cluster_points.shape[0]):
                x_2 = cluster_points[j, :].reshape(1, cluster_points[j, :].shape[0])
                result = result + kernel_distance(x_1, x_2)
        result = result / (cluster_points.shape[0] ** 2)
        return result

    term_2 = calculate_kernel_dist(np.array(clusters[i]))
    return np.array(np.repeat(term_2, data.shape[0], axis=0))


def get_kernel_clusters(i, data, clusters):
    # The function adds the terms obtained
    # using the optimization function to get cluster centers

    term_1 = get_term_1(i, data, clusters)
    term_2 = get_term_2(i, data, clusters)
    return np.add(-1 * term_1, term_2)


def plotResult(listClusterMembers, centroid):
    # Function to plot the estimated cluster centres
    # Plot cluster assignment to data

    n = listClusterMembers.__len__()
    plt.figure("result")
    plt.clf()

    for i in range(n):
        memberCluster = np.asmatrix(listClusterMembers[i])
        plt.scatter(np.ravel(memberCluster[:, 0]), np.ravel(memberCluster[:, 1]), marker=".", s=100)
        plt.scatter(np.ravel(centroid[i, 0]), np.ravel(centroid[i, 1]), marker="*", s=400, edgecolors="black")
    plt.xlabel('X coords')
    plt.ylabel('Y coords')
    plt.title(kernel + ', k=' + str(k))
    plt.show()


def kMeansKernel(data):
    # Random initialization of centres
    clusters = [[] for _ in range(k)]
    shuffled_data = data
    np.random.shuffle(shuffled_data)
    for i in range(0, data.shape[0]):
        clusters[i % k].append(data[i, :])
    n = clusters.__len__()

    iteration = 1

    while (iteration < 5):

        # centroid calculation for visualization purpose
        centroid = np.ndarray(shape=(0, data.shape[1]))
        for i in range(n):
            cluster_centroid = np.asmatrix(clusters[i]).mean(axis=0)
            centroid = np.concatenate((centroid, cluster_centroid), axis=0)

        kernel_clusters = np.ndarray(shape=(data.shape[0], 0))

        # data assignment to cluster
        for i in range(0, n):
            kernel_clusters = \
                np.concatenate((kernel_clusters,
                                get_kernel_clusters(i, data, clusters)),
                               axis=1)

        clusterMatrix = np.ravel(np.argmin(np.matrix(kernel_clusters), axis=1))

        # new data assignment
        new_clusters = [[] for _ in range(k)]
        for i in range(0, data.shape[0]):
            new_clusters[clusterMatrix[i].item()].append(data[i, :])

        clusters = new_clusters
        iteration += 1

    # Plot the estimated cluster centres
    plotResult(clusters, centroid)
    return new_clusters, centroid


if __name__ == '__main__':

    # Different datasets
    filePath1 = "data/concentric_circles.txt"
    filePath2 = "data/moons.txt"
    filePath3 = 'data/uneven_distributed_data.txt'

    X1, k1 = np.loadtxt(filePath1, delimiter=" "), 2
    X2, k2 = np.loadtxt(filePath2, delimiter=" "), 2
    X3, k3 = np.loadtxt(filePath3, delimiter=" "), 3

    X, k = X1, k1
    var = 0.02
    kernels = ['rbf_kernel', 'linear_kernel', 'quadratic_kernel']
    for kernel in kernels:
        clusterResult, centroid = kMeansKernel(X)
