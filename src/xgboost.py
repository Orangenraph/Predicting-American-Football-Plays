# src/xgboost_model.py

import pandas as pd
from xgboost import XGBClassifier


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 420,
    **hyperparams,
) -> XGBClassifier:
    """
    Train an XGBoost classifier with sensible defaults

    XGBoost is tree-based and scale-invariant, so no scaler is needed.
    Default hyperparameters are used intentionally as a strong baseline.

    Optional **hyperparams override the defaults, intended for use with
    the output of tune_xgboost() in tuning.py:
        best_params = tune_xgboost(X_train, y_train)
        model = train_xgboost(X_train, y_train, **best_params)

    Parameters
        X_train      : training features
        y_train      : training labels
        random_state : random seed for reproducibility
        **hyperparams: optional overrides (n_estimators, learning_rate, etc.)

    Returns
        model : fitted XGBClassifier
    """

    # defaults – overridden by anything passed via **hyperparams
    xgb_params = dict(
        n_estimators=800,       # how many trees are built
        learning_rate=0.02,     # how much each tree corrects the previous
        max_depth=6,            # 2^6 = 64 max leaf nodes per tree
        subsample=0.8,          # row sampling – adds entropy, reduces overfitting
        colsample_bytree=0.8,   # feature sampling – forces weak features to contribute
        eval_metric="logloss",  # binary classification metric
        random_state=random_state,
    )
    xgb_params.update(hyperparams)  # tuned params win over defaults

    model = XGBClassifier(**xgb_params)

    model.fit(X_train, y_train)
    print("[train_xgboost] Training complete.")
    return model