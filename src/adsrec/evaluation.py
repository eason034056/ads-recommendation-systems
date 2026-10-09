from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def ranking_metrics(scores: np.ndarray, targets: np.ndarray, ks: tuple[int, ...] = (10, 50)) -> dict[str, float]:
    """Compute one-positive-per-row ranking metrics without sampling negatives."""
    if scores.ndim != 2 or len(scores) != len(targets):
        raise ValueError("scores must be [examples, candidates] and match targets")
    order = np.argsort(-scores, axis=1)
    ranks = np.argmax(order == targets[:, None], axis=1) + 1
    result: dict[str, float] = {"mrr": float(np.mean(1 / ranks))}
    for k in ks:
        result[f"recall@{k}"] = float(np.mean(ranks <= k))
        result[f"ndcg@{k}"] = float(np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)))
    return result


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    if len(np.unique(labels)) != 2:
        raise ValueError("binary_metrics requires both classes")
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "logloss": float(log_loss(labels, probabilities)),
    }
