import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from kmean_pca import plotit


def plot_agglo(X_scaled, link_choice=str):
    score_s = []
    score_chs = []
    fig, axs = plt.subplots(5, 2)
    x, y = 0, 0
    for i in range(2, 11):
        Z = linkage(X_scaled, link_choice)
        y_pred = fcluster(Z=Z, t=i, criterion="maxclust")
        score_s.append(silhouette_score(X_scaled, y_pred))
        score_chs.append(calinski_harabasz_score(X_scaled, y_pred))

        axs[x, y].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred)
        axs[x, y].set_title(f"Cluster = {i}")
        if (i - 1) % 2 == 0:
            x += 1
        y = (i - 1) % 2
    fig.set_size_inches(10, 15)
    fig.suptitle(f"Linkage = {link_choice}")
    plt.subplots_adjust(top=0.85)
    fig.tight_layout(h_pad=2)

    return score_s, score_chs


if __name__ == "__main__":
    # Part 1 ==================================================================
    # generate sample data.
    X, y = make_blobs(n_samples=200, centers=4, random_state=0, cluster_std=0.5)

    # scale the data.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # plot scaled data.
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
    plt.title("Scaled blobs data with 4 centers")
    plt.tight_layout
    plt.savefig("fig/agglo_data.png")
    plt.show()

    # Part 2 ==================================================================
    link_type = ["single", "complete", "centroid", "ward", "average"]
    k_values = list(range(2, 11))
    list_s = []
    list_chs = []
    for link in link_type:
        score_s, score_chs = plot_agglo(X_scaled, link_choice=link)
        plt.savefig(f"fig/agglo_{link}.png")
        plt.show()
        list_s.append(score_s)
        list_chs.append(score_chs)

    # Part 3 ==================================================================
    fig, axs = plt.subplots(5, 2)
    i = 0
    for link in link_type:
        plotit(
            axs=axs,
            x_data=k_values,
            y_data=list_s[i],
            line="gs-",
            ylabel="score",
            name=f"Silhouette score for {link} linkage",
            row=i,
            col=0,
        )
        plotit(
            axs=axs,
            x_data=k_values,
            y_data=list_chs[i],
            line="bs-",
            ylabel="score",
            name=f"CHI score for {link} linkage",
            row=i,
            col=1,
        )
        i += 1
    fig.set_size_inches(10, 15)
    fig.suptitle("Evaluation of different linkages")
    plt.subplots_adjust(top=0.85)
    fig.tight_layout(h_pad=2)
    plt.savefig("fig/agglo_score.png")
    plt.show()

    # Part 4 ==================================================================
    for link in link_type:
        Z = linkage(X_scaled, link)
        dendrogram(Z)
        plt.title(f"{link} linkage")
        plt.savefig(f"fig/dendogram_{link}.png")
        plt.show()
