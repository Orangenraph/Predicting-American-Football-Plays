# src/data_loader.py

import os
import pandas as pd
import nflreadpy as nfl
from config import SEASONS, CACHE_PATH

def load_data() -> pd.DataFrame:
    """
    Laod NFL Play by Play data from nflready
    First call get data from nflverse and store in data,
    following calls takes in from cache instead
    """
    if os.path.exists(CACHE_PATH):
        print(f"Cache found - loading data from: {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH) 
    else:
        print(f"NO Cache found — laod seasons {SEASONS} from nflreadpy...")
        df = nfl.load_pbp(seasons=SEASONS)

        df = df.to_pandas() #convert polars df into pandas

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True) #create folder structure

        df.to_parquet(CACHE_PATH, index=False)
        print(f"Cache got successfuly stored under: {CACHE_PATH}")

    return df


def inspect_data(df: pd.DataFrame):
    """
    Show first insights of dataset
    """
    print(f"\nShape:        {df.shape[0]:,} Rows, {df.shape[1]} Columns")
    print(f"Seasons:      {sorted(df['season'].unique())}")
    print(f"Play Types:   {df['play_type'].value_counts().to_dict()}")
    print(f"Missing Entries (%):\n{(df[df.columns].isnull().mean() * 100).sort_values(ascending=False).head(10)}")


if __name__ == "__main__":
    df = load_data() 
    inspect_data(df)