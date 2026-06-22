# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

#---------------------------------------------------------------------------
# Step 1 - Filter to pass / run plays and build binary target
#---------------------------------------------------------------------------
def filter_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only pass and run plays, reset index"""
    #exclude other play types
    mask = df["play_type"].isin(["pass", "run"])
    return df[mask].copy().reset_index(drop=True)


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    encode play_type as binary target
    1 = pass, 0 = run
    """
    # convert bool evalution into 1s and 0
    return (df["play_type"] == "pass").astype(int).rename("target")


# ---------------------------------------------------------------------------
# Step 2 - Select features
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame, feature_set: list) -> pd.DataFrame:
    """return a df containing only  columns in  chosen feature set"""
    # safety check for silent error if col is missing
    missing_cols = [c for c in feature_set if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
    return df[feature_set].copy()

# ---------------------------------------------------------------------------
# Step 3 - Clean / impute missing values
# ---------------------------------------------------------------------------

def _impute_numeric(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """
    Fills missing values with median value
    Median is preffered over mean since extreme outliers can exist
    """

    # look in active feature set if it has NaNs
    numeric_features = feature_config.get("numeric", [])

    for col in numeric_features:
        if col in df.columns and df[col].isna().any():
            nan_fraction = df[col].isna().mean()
            if nan_fraction < 0.01:
                df = df.dropna(subset=[col])
            else:
                df[col] = df[col].fillna(df[col].median())
    return df


def _impute_binary(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """Fills missing values with most frequent value"""
    binary_features = feature_config.get("binary", [])

    for col in binary_features:
        if col in df.columns and df[col].isna().any():
            nan_fraction = df[col].isna().mean()
            if nan_fraction < 0.01:
                df = df.dropna(subset=[col])
            else:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals[0])
    return df


def _impute_ordinal(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """
    Fills missing ordered rankings with the median value
    Only ordinal value is down: [1,2,3,4]
    Median is used, since modus would most likely always be 1
    """
    ordinal_features = feature_config.get("ordinal", {})

    for col in ordinal_features.keys():
        if col in df.columns and df[col].isna().any():
            nan_fraction = df[col].isna().mean()
            if nan_fraction < 0.01:
                df = df.dropna(subset=[col])
            else:
                df[col] = df[col].fillna(df[col].median())
    return df

def _impute_nominal(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """Fills missing text categories with most common text category"""
    nominal_features = feature_config.get("nominal", [])

    for col in nominal_features:
        if col in df.columns and df[col].isna().any():
            nan_fraction = df[col].isna().mean()
            if nan_fraction < 0.01:
                df = df.dropna(subset=[col])
            else:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals[0])
    return df



def impute_missing(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """Apply all imputation strategies"""
    df = df.copy()
    df = _impute_numeric(df, feature_config)
    df = _impute_binary(df, feature_config)
    df = _impute_ordinal(df, feature_config)
    df = _impute_nominal(df, feature_config)
    return df


# ---------------------------------------------------------------------------
# Step 4 - Encode categorical features
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """Converts text and ordered columns into numbers for ML"""
    df = df.copy()

    # converst ordered steps into ascending numbers, 
    # only for ordinal
    #ordinal_features = feature_config.get("ordinal", {})
    #for col, categories in ordinal_features.items():
    #    if col in df.columns:
    #        enc = OrdinalEncoder(
    #            categories=[categories],
    #            handle_unknown="use_encoded_value",
    #            unknown_value=-1,
    #        )
    #        df[col] = enc.fit_transform(df[[col]])

    # convert text categories into binary columns of 0 and 1
    # only for nominals
    nominal_features = feature_config.get("nominal", [])
    nominal_present = [c for c in nominal_features if c in df.columns]
    if nominal_present:
        df = pd.get_dummies(df, columns=nominal_present, drop_first=True, dtype=int)

    return df

# ---------------------------------------------------------------------------
# Step 5 - Reporting 
# ---------------------------------------------------------------------------
 
def preprocessing_report(df_raw: pd.DataFrame, 
                        df_clean: pd.DataFrame,
                        y: pd.Series,
                        feature_set: list,
                        nan_counts_before: pd.Series,
                        nan_counts_after: pd.Series,
                        fill_values: dict) -> None:
    """
    Print a summary of the preprocessing pipeline
    
    Parameters
    ----------
    df_raw            : raw DataFrame before preprocessing
    df_clean          : cleaned DataFrame after preprocessing (WITHOUT target)
    y                 : target Series (pass/run)
    feature_set       : list of selected feature names
    nan_counts_before : Series with NaN counts per selected feature before imputation
    nan_counts_after  : Series with NaN counts per selected feature after imputation
    fill_values       : Value that fills NaN, only visual purpose 
    """
    
    print("\n" + "="*80)
    print("PREPROCESSING REPORT".center(80))
    print("="*80)
    
    # input
    print("\nINPUT:")
    print(f"   Shape: {df_raw.shape}")
    missing_pct = (df_raw.isna().sum().sum() / (df_raw.shape[0] * df_raw.shape[1])) * 100
    print(f"   Missing: {df_raw.isna().sum().sum()} NaNs ({missing_pct:.2f}%)")
    
    # output
    print("\nOUTPUT:")
    df_clean_with_target = df_clean.copy()
    df_clean_with_target["target"] = y
    print(f"   Shape: {df_clean_with_target.shape}")
    print(f"   Features (X): {df_clean.shape[1]} columns")
    print(f"   Target (y): {y.shape[0]} samples")
    print(f"   Missing: {df_clean_with_target.isna().sum().sum()} NaNs")
    
    # selected features cleaining detail
    print("\nSELECTED FEATURES IMPUTATION DETAILS:")
    print(f"   {'Feature Name':<30} | {'NaNs Before':<12} | {'NaNs After':<11} | {'Filled':<8}")
    print("   " + "-" * 70)
    # compare bfore and after cleaning
    for col in feature_set:
        before = nan_counts_before.get(col, 0)
        after = nan_counts_after.get(col, 0)
        filled = before - after
        fill_val = fill_values.get(col, "-") if before > 0 else "-"
        print(f"   {col:<30} | {before:<12,} | {after:<11,} | {filled:<8,} | {fill_val:<24}")
    
    # target
    print("\nTARGET DISTRIBUTION:")
    target_dist = y.value_counts().sort_index(ascending=False)
    
    pass_count = target_dist.get(1, 0)
    run_count = target_dist.get(0, 0)
    total = len(y)
    
    pass_pct = (pass_count / total) * 100 if total > 0 else 0
    run_pct = (run_count / total) * 100 if total > 0 else 0
    
    print(f"   1 = PASS: {pass_count:,} ({pass_pct:.1f}%)")
    print(f"   0 = RUN:  {run_count:,} ({run_pct:.1f}%)")
    
    print("\n" + "="*80 + "\n")

# ---------------------------------------------------------------------------
#  Final Step - Single entry point
# ---------------------------------------------------------------------------

def preprocess(
    df_raw: pd.DataFrame,
    feature_set: list,
    feature_config: dict,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline

    Steps
        1. Filter to pass / run plays and build binary target (1 = pass, 0 = run)
        2. Select feature columns
        3. Impute missing values
        4. Encode categorical features
        5. Run validation report and combine features into a single DataFrame

    Parameters
        df_raw         : raw play-by-play DataFrame from load_data()
        feature_set    : list of active columns to keep (e.g., FEATURE_SETS['final'])
        feature_config : dictionary mapping feature types to lists of columns

    Returns
        df_clean       : clean, processed DataFrame with features and a 'target' column
    """

    # Step 1 filter play type run or pass and generate binary predition target
    df = filter_plays(df_raw)
    y = build_target(df)

    # Step 2  extract specific columns defined in feature list
    X = select_features(df, feature_set=feature_set)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # record NaN counts of  features BEFORE imputation
    nan_counts_before = X.isna().sum()

    # calc exactly what values WILL be used for imputation - only for report! 
    fill_values = {}
    for col in feature_set:
        if col in feature_config.get("numeric", []) or col in feature_config.get("ordinal", {}):
            median_val = X[col].median()
            fill_values[col] = f"{median_val} (median)"
        elif col in feature_config.get("binary", []) or col in feature_config.get("nominal", []):
            mode_vals = X[col].mode()
            if not mode_vals.empty:
                fill_values[col] = f"{mode_vals[0]} (mode)"

    # Step 3 fill any missing values
    X_imputed = impute_missing(X, feature_config=feature_config)

    # recprd NaN counts of selected features AFTER imputation - should be 0
    nan_counts_after = X_imputed.isna().sum()

    # Step 4 transform categories to computer readable numerical values
    X_encoded = encode_features(X_imputed, feature_config=feature_config)

    # Step 5 run report
    preprocessing_report(
        df_raw=df_raw,
        df_clean=X_encoded,
        y=y,
        feature_set=feature_set,
        nan_counts_before=nan_counts_before,
        nan_counts_after=nan_counts_after,
        fill_values=fill_values
    )

    # Step 6  combine features and target into single clean df
    df_clean = X_encoded.copy()
    df_clean["target"] = y

    return df_clean

