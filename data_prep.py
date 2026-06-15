import sqlite3
import pandas as pd

DB_PATH = "musik.db"
MIN_CLASS_SIZE = 5


def migrate_year_to_int():
    """
    Konvertiert year-Werte zu INTEGER in der DB.
    SQLite ist dynamisch typisiert — CAST speichert die Werte als Integer-Affinität.
    pd.to_numeric() im load_data() stellt den korrekten Python-Typ sicher.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("UPDATE songs SET year = CAST(year AS INTEGER) WHERE year IS NOT NULL AND year != ''")
    conn.commit()

    cur.execute("SELECT year, typeof(year) FROM songs LIMIT 3")
    print("year-Stichprobe nach Migration:", cur.fetchall())
    conn.close()


def load_data(merge_small_classes=True) -> pd.DataFrame:
    """
    Lädt Songs aus der DB und bereitet sie für ML vor.

    - year wird zu int konvertiert
    - Klassen mit weniger als MIN_CLASS_SIZE Samples werden zu 'Other' zusammengefasst
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT genre, year, length FROM songs WHERE genre IS NOT NULL AND genre != ''",
        conn
    )
    conn.close()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "length"])
    df["year"] = df["year"].astype(int)

    if merge_small_classes:
        counts = df["genre"].value_counts()
        small = counts[counts < MIN_CLASS_SIZE].index
        df = df[~df["genre"].isin(small)]

    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Erstellt die Feature-Matrix X (skaliert) und den Label-Vektor y.

    Gibt zurück: X_scaled (numpy array), y (Series), feature_names (list)
    """
    from sklearn.preprocessing import StandardScaler

    genre_dummies = pd.get_dummies(df["genre"]).astype(int)
    X = pd.concat([df[["year", "length"]], genre_dummies], axis=1)
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df["genre"], feature_names


if __name__ == "__main__":
    print("Starte DB-Migration...")
    migrate_year_to_int()

    print("\nLade bereinigte Daten...")
    df = load_data()
    print(f"Songs gesamt: {len(df)}")
    print(f"Genres nach Bereinigung:\n{df['genre'].value_counts()}")
    print(f"\nJahr-Range: {df['year'].min()} – {df['year'].max()}")
    print(f"Länge-Range: {df['length'].min():.1f}s – {df['length'].max():.1f}s")

    X, y, features = build_feature_matrix(df)
    print(f"\nFeature-Matrix: {X.shape[0]} Samples × {X.shape[1]} Features")
    print(f"Features: {features}")
