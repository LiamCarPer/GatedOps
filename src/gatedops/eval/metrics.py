"""Metrics computed on the held-out split of a training run."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Return the standard classification metrics plus false alarm rate.

    The false alarm rate (false positives over all negatives) is the metric a
    predictive-maintenance or churn alarm cares about most, so it is exposed
    explicitly for gating.
    """
    true_neg, false_positives, _, true_positives = confusion_matrix(y_true, y_pred).ravel()
    negatives = true_neg + false_positives

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_alarm_rate": float(
            false_positives / negatives if negatives else 0.0
        ),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
