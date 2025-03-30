import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    # Part (a)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["target_names"] = df["target"].map(
        {0: "setosa", 1: "versicolor", 2: "virginica"}
    )
    print(df.head())
    print(df.info())

    df = df.drop("target_names", axis=1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    cov_mat = np.cov(scaled_data)
    print(f"Covariance matrix is: {cov_mat}")
    print(cov_mat.shape)

    # Part (b)
    evalues, evectors = np.linalg.eig(cov_mat)

    idx = evalues.argsort()[::-1]
    evalues = evalues[idx]
    evectors = evectors[:, idx]
    print(f"Eigenvalues of the covariance matrix are: {evalues}")
    print(f"Eigenvectors of the covariance matrix are: {evectors}")

    # Part (c)
    n_components = 2
    for n in range(2, 4):
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(scaled_data)
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance ratio for {n} components:{explained_variance}")

        # Part (d)
        X_reconstructed = pca.inverse_transform(X_pca)
        mse = mean_squared_error(scaled_data, X_reconstructed)
        print(f"MSE with {n} components: {mse}")
