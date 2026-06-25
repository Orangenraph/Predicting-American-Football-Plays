# src/models.py

import pandas as pd


def split_data(
    df_raw: pd.DataFrame,
    test_seasons: list[int] = [2022, 2023],  # at least 2 years to test
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train / test split on raw df

    Split is done BEFORE preprocessing so no future information
    leaks into the training set bc of imputation or encoding statistics

    Returns
        df_train, df_test : two raw DataFrames ready for preprocess()
    """

    if "season" not in df_raw.columns:
        raise ValueError("df_raw must contain a 'season' column for temporal splitting.")

    test_seasons_set = set(test_seasons)
    df_train = df_raw[~df_raw["season"].isin(test_seasons_set)].copy()  # everything except test
    df_test  = df_raw[ df_raw["season"].isin(test_seasons_set)].copy()

    print(
        f"[split_data] Train seasons : {sorted(df_train['season'].unique())}\n"
        f"[split_data] Test  seasons : {sorted(df_test['season'].unique())}\n"
        f"[split_data] Train rows    : {len(df_train):,}\n"
        f"[split_data] Test  rows    : {len(df_test):,}"
    )

    return df_train, df_test


def get_X_y(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a preprocessed DataFrame into features X and binary target y

    Returns
        X : feature DataFrame
        y : target Series (1 = pass, 0 = run)
    """
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]
    return X, y