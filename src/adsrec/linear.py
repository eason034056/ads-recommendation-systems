from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from .evaluation import binary_metrics


def one_hot(features: np.ndarray, cardinalities: list[int]) -> sparse.csr_matrix:
    """Sparse one-hot encoding with one block of columns per categorical field."""
    offsets = np.concatenate([[0], np.cumsum(cardinalities)[:-1]])
    rows = np.repeat(np.arange(len(features)), features.shape[1])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, (features + offsets).ravel())), shape=(len(features), int(sum(cardinalities))))


def fit_logistic(train: tuple[np.ndarray, np.ndarray], validation: tuple[np.ndarray, np.ndarray], cardinalities: list[int],
                 strengths: tuple[float, ...] = (0.01, 0.1, 1.0)) -> tuple[LogisticRegression, float]:
    """L2 logistic regression on one-hot fields; the inverse regularization strength C is chosen on validation log loss."""
    train_x, validation_x = one_hot(train[0], cardinalities), one_hot(validation[0], cardinalities)
    best: tuple[float, float, LogisticRegression] | None = None
    for strength in strengths:
        model = LogisticRegression(C=strength, max_iter=2000).fit(train_x, train[1])
        loss = binary_metrics(validation[1], model.predict_proba(validation_x)[:, 1])["logloss"]
        if best is None or loss < best[0]:
            best = (loss, strength, model)
    return best[2], best[1]
