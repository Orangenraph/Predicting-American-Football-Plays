# src/analysis.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def feature_groups(columns: list, feature_set: list, feature_config: dict) -> dict:
    """
    Map encoded columns back to their original (pre-encoding) feature

    One-hot columns of a nominal feature (e.g. all posteam_*) form one group.
    The longest matching nominal prefix wins, so posteam_type_home goes to
    posteam_type and not to posteam.

    Parameters
        columns        : encoded column names (X_test.columns)
        feature_set    : original feature names (FEATURE_SETS[...])
        feature_config : FEATURE_CONFIG, used to know which features are nominal

    Returns
        groups : dict {original feature: [encoded columns]}
    """
    nominal = [f for f in feature_config.get("nominal", []) if f in feature_set]
    groups = {}

    for col in columns:
        if col in feature_set:
            owner = col
        else:
            prefixes = [f for f in nominal if col.startswith(f + "_")]
            if not prefixes:
                raise KeyError(f"Cannot map encoded column '{col}' to a feature")
            owner = max(prefixes, key=len)
        groups.setdefault(owner, []).append(col)

    return groups


# ---------------------------------------------------------------------------
# Grouped permutation importance
# ---------------------------------------------------------------------------

def grouped_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance measured as drop in ROC AUC

    All columns of a group are shuffled with the SAME row permutation, so
    one-hot blocks stay valid rows and a team identity counts as one feature.

    Parameters
        model        : fitted model with predict_proba
        X, y         : evaluation data
        groups       : output of feature_groups()
        n_repeats    : shuffles per group
        random_state : seed for the permutations

    Returns
        df : one row per group with mean / std of the AUC drop, sorted desc.
    """
    rng = np.random.default_rng(random_state)
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []

    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            idx = rng.permutation(len(X))
            X_perm[cols] = X[cols].to_numpy()[idx]
            auc = roc_auc_score(y, model.predict_proba(X_perm)[:, 1])
            drops.append(baseline - auc)
        rows.append({
            "feature":   name,
            "n_columns": len(cols),
            "mean":      np.mean(drops),
            "std":       np.std(drops),
        })

    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    df.attrs["baseline_auc"] = baseline
    return df


# ---------------------------------------------------------------------------
# Error analysis by game situation
# ---------------------------------------------------------------------------

def bucket_report(
    y_true: pd.Series,
    y_proba: np.ndarray,
    buckets: pd.Series,
    threshold: float = 0.5,
    min_n: int = 500,
) -> pd.DataFrame:
    """
    Per-bucket performance and calibration

    calibration_gap = mean predicted pass prob - actual pass rate
    (> 0: model overestimates passing in this bucket)

    Parameters
        y_true    : true labels (1 = pass, 0 = run)
        y_proba   : predicted pass probabilities
        buckets   : bucket label per play (categorical keeps its order)
        threshold : decision threshold for accuracy
        min_n     : buckets below this size are flagged as low_n

    Returns
        df : one row per bucket
    """
    data = pd.DataFrame({
        "y":      np.asarray(y_true),
        "proba":  np.asarray(y_proba),
        "bucket": np.asarray(buckets),
    })
    if isinstance(buckets.dtype, pd.CategoricalDtype):
        data["bucket"] = pd.Categorical(data["bucket"], categories=buckets.cat.categories, ordered=True)

    rows = []
    for name, g in data.groupby("bucket", observed=True, sort=True):
        y_pred = (g["proba"] > threshold).astype(int)
        auc = roc_auc_score(g["y"], g["proba"]) if g["y"].nunique() == 2 else np.nan
        rows.append({
            "bucket":          name,
            "n":               len(g),
            "pass_rate":       g["y"].mean(),
            "mean_pred_pass":  g["proba"].mean(),
            "calibration_gap": g["proba"].mean() - g["y"].mean(),
            "accuracy":        accuracy_score(g["y"], y_pred),
            "roc_auc":         auc,
            "low_n":           len(g) < min_n,
        })

    return pd.DataFrame(rows)
