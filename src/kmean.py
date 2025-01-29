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


def assign_cluster(df, k, d, centroid):
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

def find_wcss(df, d, centroid):
    result = 0
    df_numpy = df.to_numpy()
    for x in df_numpy:
        vec_dif = x[:d] - centroid[int(x[-1])]
        result += np.dot(vec_dif.T, vec_dif)
    return result

def lloyd(df, k, max_iter=100):
    # Find the dimension of the datapoints.
    d = df.shape[1]
    # randomly assign all points to a centroid/cluster.
    df["k"] = np.random.randint(0, k, size=len(df))
    # Calculate the centroids as mean of their assigned points.
    centroid = find_centroid(df, k, d)
    # initialize while loop.
    iter = 0
    dif_centroid = np.ones((k, d))
    threshold = np.ones((k, d)) * 0
    while np.any(dif_centroid>threshold) and iter < max_iter:
        centroid_before = centroid # update centroid_result.
        iter += 1 # update number of iterations.
        # reassign each datapoint to its closest centroid.
        assign_cluster(df, k, d, centroid)
        # recalculate the centroid.
        centroid = find_centroid(df, k, d)
        # check if the centroid has moved.
        dif_centroid = np.abs(centroid_before-centroid)
    wcss = find_wcss(df, d, centroid)
    return centroid, df, iter, wcss

def macqueen(df, k, max_iter=100):
    # Find the dimension of the datapoints.
    d = df.shape[1]
    # Find number of datapoints n.
    n = df.shape[0]
    # add an empty column for cluster labels.
    df["k"] = ""
    # randomly choose k datapoints as centroids.
    # choose indices of df which are chosen to be the initial centroids.
    i_rnd = np.random.choice(n, k, replace=False)
    centroid = np.zeros((k, d))  # initialize dimensions of centroids.
    for i in range(k):
        centroid[i,:] = df.iloc[i_rnd[i], :d]
        df.at[i_rnd[i],"k"] = i # label cluster to initial centroid datapoints.
    # initialize while loop.
    iter = 0
    dif_centroid = np.ones((k, d))
    threshold = np.ones((k, d)) * 0
    while np.any(dif_centroid>threshold) and iter < max_iter:
        centroid_before = centroid # update centroid_result.
        iter += 1 # update number of iterations.
        # assign each datapoint to its closest centroid.
        assign_cluster(df, k, d, centroid)
        # recalculate the centroid.
        centroid = find_centroid(df, k, d)
        # check if the centroid has moved.
        dif_centroid = np.abs(centroid_before-centroid)
    wcss = find_wcss(df, d, centroid)
    return centroid, df, iter, wcss

def hartigan(df, k, max_iter=100):
    # Find the dimension of the datapoints.
    d = df.shape[1]
    # Find number of datapoints n.
    n = df.shape[0]
    # randomly assign all points to a centroid/cluster.
    df["k"] = np.random.randint(0, k, size=len(df))
    # Calculate the centroids as mean of their assigned points.
    centroid = find_centroid(df, k, d)
    # initialize while loop.
    iter = 0
    dif_centroid = np.ones((k, d))
    threshold = np.ones((k, d)) * 0
    while np.any(dif_centroid>threshold) and iter < max_iter:
        for x in range(n): # for each datapoint, find closest centroid.
            df_datapoint = df.iloc[x].copy().to_frame()
            assign_cluster(df_datapoint, k, d, centroid)    
        # recalculate the centroids as mean of their assigned points.
        centroid = find_centroid(df, k, d)
        centroid_before = centroid # update centroid_result.
        iter += 1 # update number of iterations.
        # assign each datapoint to its closest centroid.
        assign_cluster(df, k, d, centroid)
        # recalculate the centroid.
        centroid = find_centroid(df, k, d)
        # check if the centroid has moved.
        dif_centroid = np.abs(centroid_before-centroid)
    wcss = find_wcss(df, d, centroid)
    return centroid, df, iter, wcss


if __name__ == "__main__":
    # Set number of clusters k.
    k = 3

    # Load the dataset from scikit-learn.
    iris = load_iris()

    # Create a dataframe.
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Fix the seed for the random number generator for reproducibility.
    np.random.seed(42)

    # Lloyd Algorithm
    centroid, df, iter, wcss = lloyd(df, k)
    print("Lloyd Algorithm: ")
    #print(df)
    print(f"centroiods are: \n{centroid}")

    for x in range(k):
        print(f"{x}-cluster has {df[df["k"] == x].shape[0]} points.")
    print(f"Lloyd Algorithm took {iter} iterations, with wcss = {wcss}.")

    # Create a dataframe.
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Mac Queen Algorithm.
    centroid, df, iter, wcss = macqueen(df, k)
    print("Mac Queen Algorithm: ")
    #print(df)
    print(f"centroiods are: \n{centroid}")

    for x in range(k):
        print(f"{x}-cluster has {df[df["k"] == x].shape[0]} points.")
    print(f"Mac Queen Algorithm took {iter} iterations, with wcss = {wcss}.")

    # Create a dataframe.
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Hartigan-Wong Algorithm.
    centroid, df, iter, wcss = hartigan(df, k)
    print("Hartigan-Wong Algorithm: ")
    #print(df)
    print(f"centroiods are: \n{centroid}")

    for x in range(k):
        print(f"{x}-cluster has {df[df["k"] == x].shape[0]} points.")
    print(f"Hartigan-Wong Algorithm took {iter} iterations, with wcss = {wcss}.")

    # Elbow method
    for k in range(2,7):
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        # centroid, df, iter, wcss = lloyd(df, k)
        # centroid, df, iter, wcss = macqueen(df, k)
        centroid, df, iter, wcss = hartigan(df, k)
        print(f"{k} cluster has wcss = {wcss}")
        # print(centroid)
