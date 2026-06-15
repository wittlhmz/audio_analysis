import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_prep import load_data, build_feature_matrix

os.makedirs("plots", exist_ok=True)

GENRE_ORDER = [
    "Electronic", "Pop", "Dance", "Party",
    "Hip Hop", "Bass Music", "Schlager", "Indie",
    "Rock", "Alternative"
]


def plot_confusion_matrix(cm, labels, title: str, filename: str) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".0%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="#dddddd",
        ax=ax,
        cbar=False
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Vorhergesagtes Genre", fontsize=11, labelpad=10)
    ax.set_ylabel("Tatsächliches Genre", fontsize=11, labelpad=10)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    path = os.path.join("plots", filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {path}")


def plot_f1_comparison(results: dict, filename: str) -> None:
    names = list(results.keys())
    scores = [results[n]["macro_f1"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f0f2f5")
    ax.grid(axis="x", color="white", linewidth=0.9, zorder=0)

    colors = sns.color_palette("tab10", n_colors=len(names))
    bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.55, zorder=3)

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", ha="left", fontsize=10, fontweight="bold"
        )

    ax.set_xlim(0, max(scores) * 1.18)
    ax.set_xlabel("Macro F1-Score", fontsize=11)
    ax.set_title("Klassifikation – Vergleich der Modelle", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join("plots", filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {path}")


def plot_feature_importance(importances, feature_names: list, filename: str) -> None:
    idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in idx]
    sorted_imp = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f0f2f5")
    ax.grid(axis="x", color="white", linewidth=0.9, zorder=0)

    palette = sns.color_palette("Blues_r", n_colors=len(sorted_features))
    bars = ax.barh(sorted_features[::-1], sorted_imp[::-1],
                   color=palette[::-1], edgecolor="white", height=0.6, zorder=3)

    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Random Forest – Feature Importance", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join("plots", filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {path}")


def run_all():
    print("Lade Daten...")
    df = load_data()
    _, y, _ = build_feature_matrix(df)

    # Nur year und length als Features — Genre darf nicht rein (wäre Data Leakage)
    from sklearn.preprocessing import StandardScaler
    X_raw = df[["year", "length"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    feature_names = ["year", "length"]

    present_genres = [g for g in GENRE_ORDER if g in y.values]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    classifiers = [
        ("KNeighbors", KNeighborsClassifier(n_neighbors=7)),
        ("Linear SVC", LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)),
        ("SVC (rbf)", SVC(kernel="rbf", class_weight="balanced", random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    ]

    results = {}

    for name, clf in classifiers:
        print(f"\nTrainiere {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro_f1 = report["macro avg"]["f1-score"]
        results[name] = {"macro_f1": macro_f1}

        print(f"  Macro F1: {macro_f1:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=present_genres)
        plot_confusion_matrix(
            cm,
            labels=present_genres,
            title=f"Confusion Matrix – {name}",
            filename=f"classification_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        )

        if name == "Random Forest":
            plot_feature_importance(
                clf.feature_importances_,
                feature_names,
                "classification_feature_importance.png"
            )

    plot_f1_comparison(results, "classification_comparison.png")
    print("\nFertig. Alle Klassifikations-Plots in ./plots/")


if __name__ == "__main__":
    run_all()
