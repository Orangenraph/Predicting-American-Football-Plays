# src/logistic_regression.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 420,
    **hyperparams,
) -> Pipeline:
    """
    Train a Logistic Regression classifier with standard scaling

    LR is sensitive to feature scale, so a StandardScaler is included
    in the pipeline — ensures the scaler is fit only on training data
    and applied consistently to unseen data

    Optional **hyperparams override the defaults, intended for use with
    the output of tune_logistic_regression() in tuning.py:
        best_params = tune_logistic_regression(X_train, y_train)
        model = train_logistic_regression(X_train, y_train, **best_params)

    Parameters
        X_train      : training features
        y_train      : training labels
        max_iter     : solver iteration limit (default 1000)
        random_state : random seed for reproducibility
        **hyperparams: optional overrides (C, penalty, solver, class_weight)

    Returns
        pipeline : fitted sklearn Pipeline (scaler + LogisticRegression)
    """

    # defaults – overridden by anything passed via **hyperparams
    lr_params = dict(
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",  # handles any minor class imbalance
    )
    lr_params.update(hyperparams)  # tuned params win over defaults

    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # standardize mean to 0 and std to 1
        ("clf", LogisticRegression(**lr_params)),
    ])

    pipeline.fit(X_train, y_train)
    print("[train_logistic_regression] Training complete.")
    return pipeline