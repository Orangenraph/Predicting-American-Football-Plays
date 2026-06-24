#src/models.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def split_data(
    df_raw: pd.DataFrame,
    test_seasons: list[int] = [2022,2023], # at least 2 years to test
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train / test split on raw df

    Split is done BEFORE preprocessing for no future information
    leaks into the training set bc of imputation or encoding statistics

    Returns
        df_train, df_test : two raw DataFrames ready for preprocess()
    """
    
    # errock chcecking
    if "season" not in df_raw.columns:
        raise ValueError("df_raw must contain a 'season' column for temporal splitting.")

    # split by seasons
    test_seasons_set = set(test_seasons)
    df_train = df_raw[~df_raw["season"].isin(test_seasons_set)].copy() #everything except test
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
    split a preprocessed DataFrame into features X and binary target y

    Returns
        X : feature DataFrame
        y : target Series (1 = pass, 0 = run)
    """
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]
    return X, y


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 420,
) -> Pipeline:
    """
    Train a Logistic Regression classifier with standard scaling

    LR is sensitive to feature scale, so a StandardScaler
    is included in the pipeline, ensures the scaler is fit only on
    training data and applied consistently to unseen data

    Parameters
        X_train      : training features
        y_train      : training labels
        max_iter     : solver iteration limit (default 1000)
        random_state : random seed for reproducibility

    Returns
        pipeline : fitted sklearn Pipeline (scaler + LogisticRegression)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()), #standardizre mean to 0 and deviation to 1
        ("clf", LogisticRegression( 
            max_iter=max_iter,
            random_state=random_state,
            class_weight="balanced",   # handles any minor class imbalance
        ))
    ])

    pipeline.fit(X_train, y_train)
    print("[train_logistic_regression] Training complete.")
    return pipeline


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 420,
) -> XGBClassifier:
    """
    Train an XGBoost classifier with sensible defaults

    XGBoost is tree-based and scale-invariant, so no scaler is needed
    Default hyperparameters are used intentionally; will be separate step.

    Parameters
        X_train      : training features
        y_train      : training labels
        random_state : random seed for reproducibility

    Returns
        model : fitted XGBClassifier
    """
    model = XGBClassifier(
        n_estimators=800, # how many trees get build
        learning_rate=0.02, # how much it should get corrected
        max_depth=6, # 6 = 2^6 = 64 nodes
        subsample=0.8, # more entropy to make it robust
        colsample_bytree=0.8, # force model to also weak weak features
        use_label_encoder=False, # skip warnsignals
        eval_metric="logloss", # since it binary classiftaciton problem
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    print("[train_xgboost] Training complete.")
    return model