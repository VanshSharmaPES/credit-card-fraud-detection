"""Shared evaluation helpers for the credit-card fraud project.

Threshold selection uses validation labels only. Test labels are used only once,
for final reporting. Probability metrics compare predicted fraud probabilities
with the binary target because the project brief explicitly requests them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    roc_auc_score,
)


def find_threshold(y_val: Iterable[int], p_val: Iterable[float]) -> float:
    """Return the F1-maximizing probability threshold, selected on validation."""
    y = np.asarray(y_val, dtype=int).reshape(-1)
    p = np.asarray(p_val, dtype=float).reshape(-1)
    _validate_binary_probabilities(y, p)
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        return 0.5
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1], denom,
        out=np.zeros_like(denom), where=denom != 0,
    )
    best = np.flatnonzero(np.isclose(f1, np.max(f1)))
    # Resolve ties toward the higher threshold to reduce avoidable false alarms.
    return float(thresholds[best[-1]])


def classification_metrics(y: Iterable[int], p: Iterable[float], threshold: float) -> dict:
    """Compute thresholded classification measures and ranking measures."""
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    _validate_binary_probabilities(y, p)
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float((tp / (tp + fn)) if (tp + fn) else 0.0),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def regression_metrics(y: Iterable[int], p: Iterable[float]) -> dict:
    """Compute requested RMSE, MAE, R2 on probabilities versus binary labels."""
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if y.size != p.size or y.size == 0:
        raise ValueError("y and p must have the same non-zero length")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if y.size > 1 else None,
    }


def evaluate_model(
    name: str,
    y_val: Iterable[int], p_val: Iterable[float],
    y_test: Iterable[int], p_test: Iterable[float],
    row_id_val: Iterable, row_id_test: Iterable,
    train_seconds: float, predict_seconds: float, notes: str = "",
    hardware: str = "unspecified",
    output_dir: str | Path = ".",
    folder: str | None = None,
) -> dict:
    """Choose threshold on validation, score test, and save JSON/CSV artifacts."""
    yv, pv = np.asarray(y_val, int).reshape(-1), np.asarray(p_val, float).reshape(-1)
    yt, pt = np.asarray(y_test, int).reshape(-1), np.asarray(p_test, float).reshape(-1)
    rid_v, rid_t = np.asarray(row_id_val).reshape(-1), np.asarray(row_id_test).reshape(-1)
    if not (len(yv) == len(pv) == len(rid_v)):
        raise ValueError("Validation labels, probabilities, and row IDs must align")
    if not (len(yt) == len(pt) == len(rid_t)):
        raise ValueError("Test labels, probabilities, and row IDs must align")
    threshold = find_threshold(yv, pv)
    result = {
        "framework": str(name),
        **classification_metrics(yt, pt, threshold),
        **regression_metrics(yt, pt),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "hardware": str(hardware),
        "notes": str(notes),
    }
    root = Path(output_dir)
    result_dir = root / "results" / folder if folder else root / "results"
    pred_dir = root / "preds" / folder if folder else root / "preds"
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    pd.DataFrame({"row_id": rid_v, "y_true": yv, "y_prob": pv}).to_csv(
        pred_dir / f"{name}_val.csv", index=False
    )
    pd.DataFrame({"row_id": rid_t, "y_true": yt, "y_prob": pt}).to_csv(
        pred_dir / f"{name}_test.csv", index=False
    )
    return result


def permutation_importance(
    predict_fn: Callable[[pd.DataFrame], Iterable[float]],
    X_val: pd.DataFrame,
    y_val: Iterable[int],
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank columns by validation average-precision loss after permutation."""
    X = X_val.copy() if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val)
    y = np.asarray(y_val, dtype=int).reshape(-1)
    if len(X) != len(y):
        raise ValueError("X_val and y_val must have the same number of rows")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    baseline = average_precision_score(y, np.asarray(predict_fn(X), float).reshape(-1))
    rng = np.random.default_rng(random_state)
    records = []
    for column in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = average_precision_score(
                y, np.asarray(predict_fn(shuffled), float).reshape(-1)
            )
            drops.append(float(baseline - score))
        records.append({
            "feature": str(column),
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops, ddof=0)),
            "baseline_pr_auc": float(baseline),
        })
    return pd.DataFrame(records).sort_values(
        "importance_mean", ascending=False, ignore_index=True
    )


def _validate_binary_probabilities(y: np.ndarray, p: np.ndarray) -> None:
    if y.size != p.size or y.size == 0:
        raise ValueError("y and p must have the same non-zero length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only binary labels 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p must contain finite probabilities in [0, 1]")
