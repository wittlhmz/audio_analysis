import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from data_prep import load_data, build_feature_matrix

os.makedirs("plots", exist_ok=True)

PALETTE = sns.color_palette("tab10", n_colors=10)


def make_cluster_plot(X_2d: np.ndarray, labels, title: str, filename: str, n_clusters: int) -> None:
    unique = sorted(set(labels))
    colors = sns.color_palette("tab10", n_colors=len(unique))
    color_map = {label: colors[i] for i, label in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f0f2f5")
    ax.grid(color="white", linewidth=0.8, zorder=0)

    for label in unique:
        mask = labels == label
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color_map[label]],
            label=f"Cluster {label}",
            alpha=0.75,
            s=40,
            edgecolors="white",
            linewidths=0.3,
            zorder=3
        )

    score = silhouette_score(X_2d, labels)
    subtitle = f"k={n_clusters}  |  Silhouette-Score: {score:.3f}"

    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha="center", fontsize=10, color="#555555")
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        title_fontsize=10
    )

    plt.tight_layout()
    path = os.path.join("plots", filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {path}  (k={n_clusters}, Silhouette={score:.3f})")


def run_all():
    print("Lade Daten...")
    df = load_data()
    X_scaled, _, _ = build_feature_matrix(df)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    # KMeans
    print("Berechne KMeans...")
    km = KMeans(n_clusters=7, random_state=42, n_init=10)
    labels_km = km.fit_predict(X_scaled)
    make_cluster_plot(X_2d, labels_km, "KMeans", "clustering_kmeans.png", n_clusters=7)

    # MeanShift
    print("Berechne MeanShift (bandwidth-Schätzung läuft...)...")
    bw = estimate_bandwidth(X_scaled, quantile=0.15, n_samples=500, random_state=42)
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    labels_ms = ms.fit_predict(X_scaled)
    n_ms = len(set(labels_ms))
    make_cluster_plot(X_2d, labels_ms, "MeanShift", "clustering_meanshift.png", n_clusters=n_ms)

    # GMM
    print("Berechne GMM (Gaussian Mixture Model)...")
    gmm = GaussianMixture(n_components=7, random_state=42)
    labels_gmm = gmm.fit_predict(X_scaled)
    make_cluster_plot(X_2d, labels_gmm, "GMM – Gaussian Mixture Model", "clustering_gmm.png", n_clusters=7)

    # Spectral Clustering
    print("Berechne Spectral Clustering...")
    sc = SpectralClustering(n_clusters=7, random_state=42, affinity="nearest_neighbors", n_neighbors=10)
    labels_sc = sc.fit_predict(X_scaled)
    make_cluster_plot(X_2d, labels_sc, "Spectral Clustering", "clustering_spectral.png", n_clusters=7)

    print("\nFertig. Alle Cluster-Plots in ./plots/")


if __name__ == "__main__":
    run_all()
