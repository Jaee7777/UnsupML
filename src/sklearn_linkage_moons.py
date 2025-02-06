import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn_linkage_blobs import plot_agglo, plot_score


if __name__ == "__main__":
    # Part 1 ==================================================================
    # generate sample data.
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    # scale the data.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # plot scaled data.
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
    plt.title("Scaled moons data")
    plt.tight_layout
    plt.savefig("fig/agglo_data_c.png")
    plt.show()

    # Part 2 ==================================================================
    link_type = ["single", "complete", "ward", "average"]
    k_values = list(range(2, 11))
    list_s = []
    list_chs = []
    for link in link_type:
        score_s, score_chs = plot_agglo(X_scaled, link_choice=link)
        plt.savefig(f"fig/agglo_{link}_c.png")
        plt.show()
        list_s.append(score_s)
        list_chs.append(score_chs)

    # Part 3 ==================================================================
    i = 0
    for link in link_type:
        i += 1
        plt.subplot(4, 2, 2 * i - 1)
        plot_score(k_values, list_s[i - 1], "gs-")
        plt.title(f"Silhouette score for {link} linkage")
        # plt.ylim(0, 0.8)
        plt.subplot(4, 2, 2 * i)
        plot_score(k_values, list_chs[i - 1], "bs-")
        plt.title(f"CHI score for {link} linkage")
        # plt.ylim(0, 5000)
    plt.gcf().set_size_inches(10, 12)
    plt.savefig("fig/agglo_score_c.png")
    plt.show()

    # Part 4 ==================================================================
    for link in link_type:
        Z = linkage(X_scaled, link)
        dendrogram(Z)
        plt.title(f"{link} linkage")
        plt.savefig(f"fig/dendogram_{link}_c.png")
        plt.show()
