import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def find_centroid(df, k, d):
    centroid = np.zeros((k, d))  # initialize dimensions of centroids.
    for i in range(k):  # search for clusters.
        for j in range(d):  # search for dimensions.
            # find mean for jth dimension of the datapoints in k = i cluster.
            centroid[i, j] = df[df["k"] == i].iloc[:, j].mean()
    return centroid


def assign_cluster(df, k, centroid):
    # initialize a temparory space to store Euclidean distances.
    temp = np.zeros(k)
    for x in range(df.shape[0]):
        # locate a datapoint in numpy format.
        point = df.iloc[x, 0:d].to_numpy()
        for y in range(k):
            diff = point - centroid[y, :]  # vector subtraction.
            temp[y] = np.dot(diff.T, diff)  # dot product of the vector subtraction.
        # assign the closest centroid/cluster to the data point.
        df.iloc[x, -1] = np.argmin(temp)
    return


def lloyd(df, k, max_iter=50):
    # Find the dimension of the datapoints.
    d = df.shape[1]

    # randomly assign all points to a centroid/cluster.
    df["k"] = np.random.randint(0, k, size=len(df))
    print(df)

    # Calculate the centroids as mean of their assigned points.
    centroid = find_centroid(df, k, d)
    for _ in range(max_iter):
        # reassign each datapoint to its closest centroid.
        assign_cluster(df, k, centroid)
        # recalculate the centroid.
        centroid = find_centroid(df, k, d)
    return centroid, df


def macqueen(X, k, max_iters=100):
    """MacQueen's k-means algorithm."""

    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the closest centroid
        assignments = np.array(
            [np.argmin(np.linalg.norm(X - c, axis=1)) for c in centroids]
        )

        # Update centroids
        new_centroids = np.array([X[assignments == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return assignments, centroids


def hartigan(k, data):
    pass


if __name__ == "__main__":
    # Set number of clusters k.
    k = 3

    # Load the dataset from scikit-learn.
    iris = load_iris()

    # Create a dataframe.
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # df["target"] = iris.target

    # Find the dimension of the datapoints.
    d = df.shape[1]

    # Fix the seed for the random number generator for reproducibility.
    np.random.seed(42)

    centroid, df = lloyd(df, k)
    print("Lloyd Algorithm: ")
    print(df)
    print(f"centroiods are: \n{centroid}")

    for x in range(k):
        print(f"{x}-cluster has {df[df["k"] == x].shape[0]} points.")


"""
    centroids, assignments = lloyd_algorithm(X, k)

    print("Centroids:", centroids)
    print("Assignments:", assignments)

    centroids, assignments = macqueen(X, k)

    print("Centroids:", centroids)
    print("Assignments:", assignments)
"""
