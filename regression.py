import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data_prep import load_data

os.makedirs("plots", exist_ok=True)


def build_regression_features(df):
    """Genre + Jahr → Länge vorhersagen. Kein Leakage da Ziel = Länge."""
    genre_dummies = pd.get_dummies(df["genre"]).astype(int)
    X = pd.concat([df[["year"]], genre_dummies], axis=1)
    y = df["length"].values
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_names


def plot_pred_vs_actual(y_test, y_pred, title: str, filename: str,
                        r2: float, mae: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f0f2f5")
    ax.grid(color="white", linewidth=0.9, zorder=0)

    ax.scatter(y_test, y_pred, alpha=0.5, s=30, color="#4c72b0",
               edgecolors="white", linewidths=0.3, zorder=3)

    lims = [min(y_test.min(), y_pred.min()) - 10,
            max(y_test.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, color="#e05252", linewidth=1.5,
            linestyle="--", zorder=4, label="Ideale Vorhersage")

    ax.set_xlabel("Tatsächliche Länge (s)", fontsize=11)
    ax.set_ylabel("Vorhergesagte Länge (s)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.text(0.04, 0.93, f"R² = {r2:.3f}    MAE = {mae:.1f} s",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    ax.legend(fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join("plots", filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {path}  (R²={r2:.3f}, MAE={mae:.1f}s)")


def plot_r2_comparison(results: dict, filename: str) -> None:
    names = list(results.keys())
    r2s = [results[n]["r2"] for n in names]
    maes = [results[n]["mae"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("Regression – Modellvergleich", fontsize=14,
                 fontweight="bold", y=1.01)

    palette = sns.color_palette("tab10", n_colors=len(names))

    for ax, values, label, fmt in zip(
        axes,
        [r2s, maes],
        ["R²-Score (höher = besser)", "MAE in Sekunden (niedriger = besser)"],
        [".3f", ".1f"]
    ):
        ax.set_facecolor("#f0f2f5")
        ax.grid(axis="x", color="white", linewidth=0.9, zorder=0)
        bars = ax.barh(names, values, color=palette,
                       edgecolor="white", height=0.5, zorder=3)
        for bar, v in zip(bars, values):
            ax.text(v + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:{fmt}}", va="center", fontsize=10, fontweight="bold")
        ax.set_xlabel(label, fontsize=10)
        ax.set_xlim(0, max(values) * 1.2)
        ax.tick_params(labelsize=10)
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
    X_scaled, y, feature_names = build_regression_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Ziel (Länge): Mittelwert={y.mean():.1f}s, Std={y.std():.1f}s")

    regressors = [
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=1.0, max_iter=5000)),
        ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)),
        ("SVR (linear)", SVR(kernel="linear", C=1.0)),
    ]

    results = {}

    for name, reg in regressors:
        print(f"\nTrainiere {name}...")
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {"r2": r2, "mae": mae}

        plot_pred_vs_actual(
            y_test, y_pred,
            title=f"Predicted vs. Actual – {name}",
            filename=f"regression_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png",
            r2=r2, mae=mae
        )

    plot_r2_comparison(results, "regression_comparison.png")
    print("\nFertig. Alle Regressions-Plots in ./plots/")


if __name__ == "__main__":
    run_all()
