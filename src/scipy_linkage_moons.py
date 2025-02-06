import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from scipy_linkage_blobs import plot_agglo
from scipy.cluster.hierarchy import dendrogram, linkage
from kmean_pca import plotit


if __name__ == "__main__":
    # Part 1 ==================================================================
    # generate sample data.
    X, y = make_moons(n_samples=200, random_state=0, noise=0.2)

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
