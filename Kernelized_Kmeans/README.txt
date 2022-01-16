kmeans.py: Implementation of Lloyd's method to cluster data.

kernel_kmeans.py: Implementation of Kernelized Lloyd's method to cluster data.

Datasets:
concentric_circles.txt: Contains data in R2 space. Data present in concentric circles. 2 clusters
moons.txt: Contains data in R2 space. Data present in moon format. 2 clusters
uneven_distributed_data.txt: Contains data in R2 space. Spaced data with uneven distribution. 3 clusters

Results:
The rbf kernel is successfully able to identify the clusters when data is present in concentric circles.
It works similar to clustering without kernel for linear kernel.
Quadratic kernel does not perform better than rbf kernel.
All the kernels fail to cluster the moon dataset properly. They give similar results to clustering without kernelization.
The rbf kernel is able to produce better results by identifying a cluster among 3 in the dataset.
Quadratic kernel is also able to produce better results by identifying a cluster among 3 in the dataset. Linear kernel performs the worst.