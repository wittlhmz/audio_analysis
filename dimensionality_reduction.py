import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding

from data_prep import load_data, build_feature_matrix

os.makedirs("plots", exist_ok=True)

GENRE_ORDER = [
    "Electronic", "Pop", "Dance", "Party",
    "Hip Hop", "Bass Music", "Schlager", "Indie",
    "Rock", "Alternative"
]
PALETTE = sns.color_palette("tab10", n_colors=len(GENRE_ORDER))
COLOR_MAP = dict(zip(GENRE_ORDER, PALETTE))


def make_plot(X_2d: np.ndarray, genres, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f0f2f5")
    ax.grid(color="white", linewidth=0.8, zorder=0)

    for genre in GENRE_ORDER:
        mask = genres == genre
        if mask.any():
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[COLOR_MAP[genre]],
                label=genre,
                alpha=0.75,
                s=40,
                edgecolors="white",
                linewidths=0.3,
                zorder=3
            )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Komponente 1", fontsize=11)
    ax.set_ylabel("Komponente 2", fontsize=11)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        title="Genre",
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
    print(f"  Gespeichert: {path}")


def run_all():
    print("Lade Daten...")
    df = load_data()
    X_scaled, genres, _ = build_feature_matrix(df)
    genres = genres.reset_index(drop=True)

    methods = [
        (
            "PCA",
            lambda X: PCA(n_components=2, random_state=42).fit_transform(X),
            "pca.png"
        ),
        (
            "Randomized PCA",
            lambda X: PCA(n_components=2, svd_solver="randomized", random_state=42).fit_transform(X),
            "pca_randomized.png"
        ),
        (
            "t-SNE",
            lambda X: TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(X),
            "tsne.png"
        ),
        (
            "Isomap",
            lambda X: Isomap(n_neighbors=20, n_components=2).fit_transform(X),
            "isomap.png"
        ),
        (
            "Spectral Embedding",
            lambda X: SpectralEmbedding(n_components=2, random_state=42).fit_transform(X),
            "spectral_embedding.png"
        ),
        (
            "LLE (Locally Linear Embedding)",
            lambda X: LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42).fit_transform(X),
            "lle.png"
        ),
    ]

    for name, transform, filename in methods:
        print(f"Berechne {name}...")
        X_2d = transform(X_scaled)
        make_plot(X_2d, genres, name, filename)

    print("\nFertig. Alle Plots in ./plots/")


if __name__ == "__main__":
    run_all()
