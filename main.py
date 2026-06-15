import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from data_prep import load_data, build_feature_matrix

if __name__ == "__main__":
    df = load_data()
    X_scaled, y, feature_names = build_feature_matrix(df)

    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=df["cluster"].astype(str),
        hover_data=[df["genre"], df["year"], df["length"]],
        title="Musik-Cluster (KMeans + PCA)"
    )
    fig.show()

    print("\nSilhouette-Scores:")
    for k in range(2, 16):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  k={k}: {score:.3f}")
