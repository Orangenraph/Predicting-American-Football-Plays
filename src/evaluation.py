#src/evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

'''Self modules'''
from config import FIGURES_MODELS, RESULTS_METRICS, PLOT_DPI


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    feature_set: str,
) -> dict:
    """
    Compute classification metrics for fitted model and append to metrics.csv.

    Metrics computed
        accuracy  : overall share of correct predictions
        precision : weighted precision
        recall    : weighted recall
        f1        : weighted F1-score (accounts for class imbalance)
        roc_auc   : area under the ROC curve

    Parameters
        model        : fitted model
        X_test       : test feature df
        y_test       : true binary labels (1 = pass, 0 = run)
        model_name   : label used in metrics.csv  (e.g. "LogisticRegression")
        feature_set  : feature set key used in metrics.csv (e.g. "final")

    Returns
        metrics : dict with all computed metric values
    """
    y_pred  = model.predict(X_test)# for accuracy, precision recall, and f1
    y_proba = model.predict_proba(X_test)[:, 1] #odds for a 0 and 1, needed for roc_auc

    # configure metadata
    metrics = {
        "model":       model_name,
        "feature_set": feature_set,
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "precision":   round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall":      round(recall_score(y_test, y_pred, average="weighted"), 4),
        "f1":          round(f1_score(y_test, y_pred, average="weighted"), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_proba), 4),
    }

    _append_metrics(metrics)

    print(
        f"\n[evaluate_model] {model_name} | feature_set={feature_set}\n"
        f"  Accuracy  : {metrics['accuracy']}\n"
        f"  F1        : {metrics['f1']}\n"
        f"  ROC-AUC   : {metrics['roc_auc']}\n"
        f"  Precision : {metrics['precision']}\n"
        f"  Recall    : {metrics['recall']}\n"
    )

    return metrics


def _append_metrics(metrics: dict) -> None:
    """
    Append a single metrics dict as a new row in metrics.csv
    creates the with headers if it does not yet exist
    overwrites any existing row with the same (model, feature_set) combination
    """
    RESULTS_METRICS.parent.mkdir(parents=True, exist_ok=True) #check dic path

    new_row = pd.DataFrame([metrics]) #convert into single row df

    # update metrics file, if not exist init
    if RESULTS_METRICS.exists():
        df_existing = pd.read_csv(RESULTS_METRICS)
        # drop old row for same model + feature_set if it exists
        df_existing = df_existing[
            ~(
                (df_existing["model"] == metrics["model"]) &
                (df_existing["feature_set"] == metrics["feature_set"])
            )
        ]
        #append new metrics to cleaned existing data
        df_updated = pd.concat([df_existing, new_row], ignore_index=True)
    else:
        # create new df if file not exist
        df_updated = new_row

    # safe
    df_updated.to_csv(RESULTS_METRICS, index=False)
    print(f"[evaluate_model] Metrics saved in {RESULTS_METRICS}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    feature_set: str,
) -> None:
    """
    Plot and save a normalised confusion matrix heatmap

    The matrix is normalised by true label (rows) so each cell shows 
    share of plays in class predicted as pass or run

    Parameters
        model        : fitted model
        X_test       : test feature DataFrame
        y_test       : true binary labels
        model_name   : used in the plot title and filename
        feature_set  : used in the plot title and filename
    """
    FIGURES_MODELS.mkdir(parents=True, exist_ok=True) #check dic path

    # generate predictions and compute confusion matix
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, normalize="true")

    # set up plot 
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Run (0)", "Pass (1)"],
        yticklabels=["Run (0)", "Pass (1)"],
        ax=ax,
    )

    # set labels + title
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {model_name} ({feature_set})")
    fig.tight_layout()

    # save 
    save_path = FIGURES_MODELS / f"cm_{model_name}_{feature_set}.png"
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"[plot_confusion_matrix] Saved in {save_path}")


def plot_roc_curve(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_set: str,
) -> None:
    """
    Plot and save ROC curves for one or more models on same axes

    Passing multiple models overlay LR, XGBoost, and NN in one
    figure
    
    Parameters
        models      : fitted model                  
        X_test      : test feature df
        y_test      : true binary labels
        feature_set : used in the filename and title
    """
    FIGURES_MODELS.mkdir(parents=True, exist_ok=True) #check path

    fig, ax = plt.subplots(figsize=(6, 5))

    #calc and plot ROC curve for each model
    for name, model in models.items():
        # get class probabilities and calc true false positives rates
        y_proba  = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc      = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    # plot random baseline 50% line for refrence
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — feature set: {feature_set}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    
    # save
    save_path = FIGURES_MODELS / f"roc_{feature_set}.png"
    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"[plot_roc_curve] Saved in {save_path}")


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    feature_set: str | None = None,
) -> pd.DataFrame:
    """
    Load metrics.csv and return a formatted summary df

    Optionally filter by feature_set to compare all models on one feature set,
    or leave None to see the full results table.
    """
    
    # check for file first
    if not RESULTS_METRICS.exists():
        raise FileNotFoundError(
            f"No metrics file found at {RESULTS_METRICS}. "
            "Run evaluate_model() for at least one model first."
        )

    df = pd.read_csv(RESULTS_METRICS)

    # check for features
    if feature_set is not None:
        df = df[df["feature_set"] == feature_set]

    # sort results by feature set and performance desc.
    df_summary = (
        df
        .sort_values(["feature_set", "roc_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return df_summary