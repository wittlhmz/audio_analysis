import dataReader
import pandas as pd
import sqlite3

if __name__ == "__main__":
    conn = sqlite3.connect("musik.db")
    query = """
    SELECT *
    FROM songs
    WHERE artist = 'Westernhagen'
    """

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    df = pd.read_sql(query, conn)
    print(df)
    conn.close()
