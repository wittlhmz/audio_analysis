from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
from pathlib import Path
from tabulate import tabulate
import pandas as pd



def extract_mp3_metadata(mp3_path):
    try:
        audio = EasyID3(mp3_path)
        audio_info = MP3(mp3_path)

        metadata = {
            "file_path": str(mp3_path),
            "title": audio.get("title", [""])[0],
            "artist": audio.get("artist", [""])[0],
            "album": audio.get("album", [""])[0],
            "genre": audio.get("genre", [""])[0],
            "year": audio.get("date", [""])[0],
            "length": audio_info.info.length
        }

        return metadata

    except Exception as e:
        print(f"Fehler bei {mp3_path}: {e}")
        return None


def read_music_folder(folder_path):
    metadata_list = []
    root = Path(folder_path)

    for mp3_path in root.rglob("*.mp3"):
        metadata = extract_mp3_metadata(mp3_path)
        if metadata:
            metadata_list.append(metadata)

    df = pd.DataFrame(metadata_list, columns=[
        "file_path",
        "title",
        "artist",
        "album",
        "genre",
        "year",
        "length"
    ])
    return df


def readData():
    """Ließt die Metadaten und erstellt eine csv-Datei"""
    folder_path = r"C:\Users\marius\Desktop\Musik"
    df = read_music_folder(folder_path)
    df.to_csv("musikbibliothek.csv", index=False)
    print("Metadaten wurden gespeichert.")


def printData():
    """Druckt die Tabelle in die Konsole"""
    df = pd.read_csv("musikbibliothek.csv")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    df["length_min"] = (df["length"] / 60).round(2)
    print(tabulate(df, headers="keys", tablefmt="psql"))


def sumGenre():
    df = pd.read_csv("musikbibliothek.csv")
    genre_counts = df["genre"].value_counts()
    print(genre_counts)
