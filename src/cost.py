"""Money-based decision costs for the fraud review workflow.

Amounts and review fees must use the same currency unit. A flagged transaction
incurs one review fee; a missed fraud incurs its raw transaction amount.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _validated_arrays(
    y_true: Iterable[int],
    y_prob: Iterable[float] | None = None,
    amount: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    raw_y = np.asarray(y_true).reshape(-1)
    try:
        numeric_y = raw_y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must contain only binary labels 0 and 1") from exc
    if numeric_y.size == 0 or not np.isfinite(numeric_y).all() or not np.isin(numeric_y, [0, 1]).all():
        raise ValueError("y_true must be a non-empty array of binary labels")
    y = numeric_y.astype(int)
    amounts = np.asarray(amount, dtype=float).reshape(-1)
    if amounts.size != y.size:
        raise ValueError("amount and y_true must have the same length")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("amount must contain finite, non-negative values")
    if y_prob is None:
        return y, None, amounts
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if p.size != y.size:
        raise ValueError("y_prob and y_true must have the same length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    return y, p, amounts


def cost_breakdown(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    amount: Iterable[float],
    threshold: float,
    fee: float,
) -> dict[str, float | int]:
    """Return review, missed-fraud, and total costs at a probability threshold."""
    y, p, amounts = _validated_arrays(y_true, y_prob, amount)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    fee = float(fee)
    if not np.isfinite(fee) or fee < 0:
        raise ValueError("fee must be finite and non-negative")

    flagged = p >= float(threshold)
    missed = (y == 1) & ~flagged
    review_count = int(flagged.sum())
    missed_fraud_amount = float(amounts[missed].sum())
    review_cost = float(review_count * fee)
    baseline = float(amounts[y == 1].sum())
    return {
        "threshold": float(threshold),
        "review_fee": fee,
        "frauds_caught": int(((y == 1) & flagged).sum()),
        "false_alarms": int(((y == 0) & flagged).sum()),
        "reviews": review_count,
        "missed_fraud_amount": missed_fraud_amount,
        "review_cost": review_cost,
        "total_cost": missed_fraud_amount + review_cost,
        "no_model_cost": baseline,
        "money_saved": baseline - missed_fraud_amount - review_cost,
    }


def total_cost(
    y: Iterable[int], p: Iterable[float], amount: Iterable[float],
    threshold: float, fee: float,
) -> float:
    """Return missed-fraud amount plus the review fee for every alert."""
    return float(cost_breakdown(y, p, amount, threshold, fee)["total_cost"])


def no_model_cost(y: Iterable[int], amount: Iterable[float]) -> float:
    """Return the fraud amount lost when no transactions are reviewed."""
    labels, _, amounts = _validated_arrays(y, amount=amount)
    return float(amounts[labels == 1].sum())


def threshold_cost_curve(
    y: Iterable[int], p: Iterable[float], amount: Iterable[float], fee: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all distinct alert thresholds and their costs in O(n log n)."""
    labels, probs, amounts = _validated_arrays(y, p, amount)
    fee = float(fee)
    if not np.isfinite(fee) or fee < 0:
        raise ValueError("fee must be finite and non-negative")

    order = np.argsort(-probs, kind="mergesort")
    sorted_probs = probs[order]
    sorted_fraud_amounts = amounts[order] * (labels[order] == 1)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_probs)) + 1, len(sorted_probs)]
    thresholds = sorted_probs[group_ends - 1]
    flagged_counts = group_ends.astype(float)
    caught_amounts = np.cumsum(sorted_fraud_amounts)[group_ends - 1]
    costs = float(amounts[labels == 1].sum()) - caught_amounts + fee * flagged_counts

    # A threshold above the maximum probability represents reviewing no rows.
    thresholds = np.r_[thresholds, np.nextafter(float(np.max(probs)), np.inf)]
    costs = np.r_[costs, float(amounts[labels == 1].sum())]
    return thresholds, costs


def best_cost_threshold(
    y_val: Iterable[int], p_val: Iterable[float], amount_val: Iterable[float], fee: float,
) -> float:
    """Choose the validation threshold with minimum assumed monetary cost.

    Equal-cost thresholds prefer fewer reviews, then the higher threshold.
    Test labels and test amounts must never be passed to this function.
    """
    thresholds, costs = threshold_cost_curve(y_val, p_val, amount_val, fee)
    _, probs, _ = _validated_arrays(y_val, p_val, amount_val)
    order = np.argsort(-probs, kind="mergesort")
    sorted_probs = probs[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_probs)) + 1, len(sorted_probs)]
    alerts = np.r_[group_ends, 0]
    best = np.lexsort((-thresholds, alerts, costs))[0]
    return float(thresholds[best])
