import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def plot_agglo(X_scaled, link_choice=str):
    score_s = []
    score_chs = []
    for i in range(2, 11):
        Z = linkage(X_scaled, link_choice)
        y_pred = fcluster(Z=Z, t=i, criterion="maxclust")
        score_s.append(silhouette_score(X_scaled, y_pred))
        score_chs.append(calinski_harabasz_score(X_scaled, y_pred))

        plt.subplot(5, 2, i - 1)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred)
        plt.title(f"Linkage = {link_choice} & Cluster = {i}")
    plt.gcf().set_size_inches(10, 15)
    plt.tight_layout
    return score_s, score_chs


def plot_score(k, score, color):
    plt.plot(k, score, color)
    plt.xlabel("k")
    plt.ylabel("score")
    plt.grid(True)
    plt.tight_layout


if __name__ == "__main__":
    # Part 1 ==================================================================
    # generate sample data.
    X, y = make_blobs(n_samples=1000, centers=4, random_state=0, cluster_std=0.5)

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
    i = 0
    for link in link_type:
        i += 1
        plt.subplot(5, 2, 2 * i - 1)
        plot_score(k_values, list_s[i - 1], "gs-")
        plt.title(f"Silhouette score for {link} linkage")
        # plt.ylim(0, 0.8)
        plt.subplot(5, 2, 2 * i)
        plot_score(k_values, list_chs[i - 1], "bs-")
        plt.title(f"CHI score for {link} linkage")
        # plt.ylim(0, 5000)
    plt.gcf().set_size_inches(10, 15)
    plt.savefig("fig/agglo_score.png")
    plt.show()

    # Part 4 ==================================================================
    for link in link_type:
        Z = linkage(X_scaled, link)
        dendrogram(Z)
        plt.title(f"{link} linkage")
        plt.savefig(f"fig/dendogram_{link}.png")
        plt.show()
