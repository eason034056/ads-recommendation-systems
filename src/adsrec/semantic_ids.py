from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def cooccurrence_vectors(history: np.ndarray, target: np.ndarray, item_count: int, dimension: int, seed: int) -> np.ndarray:
    """Item vectors from a truncated SVD of positive PMI between each training target and its history.

    Items with no co-occurrence get zero vectors, so they share one semantic-ID prefix and are
    told apart only by the deduplication token.
    """
    rows, columns = np.repeat(target, history.shape[1]), history.ravel()
    counts = sparse.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(item_count, item_count)).tocsr()
    counts = (counts + counts.T).tocoo()
    total, item_totals = counts.sum(), np.asarray(counts.sum(axis=1)).ravel()
    # PMI compares how often two items co-occur with how often independence would predict.
    pmi = np.log(counts.data * total / (item_totals[counts.row] * item_totals[counts.col]))
    ppmi = sparse.csr_matrix((np.maximum(pmi, 0), (counts.row, counts.col)), shape=counts.shape)
    return TruncatedSVD(dimension, random_state=seed).fit_transform(ppmi)[1:]


def deduplicate(codes: np.ndarray, popularity: np.ndarray) -> np.ndarray:
    """Append a final token that makes every item's semantic ID unique.

    Items sharing all residual codes are numbered by descending training popularity, so the
    token means the same thing under every prefix: 0 is that prefix's most popular item.
    """
    frame = pd.DataFrame(codes).assign(popularity=popularity)
    ranked = frame.sort_values("popularity", ascending=False, kind="stable")
    suffix = ranked.groupby(list(range(codes.shape[1])), sort=False).cumcount().reindex(frame.index)
    return np.column_stack([codes, suffix.to_numpy()])


def collision_stats(codes: np.ndarray) -> dict[str, float]:
    """Describe how residual codes (before deduplication) spread items over semantic IDs."""
    _, sizes = np.unique(codes, axis=0, return_counts=True)
    return {
        "items": int(len(codes)),
        "distinct_ids": int(len(sizes)),
        "largest_group": int(sizes.max()),
        "items_sharing_an_id": float(sizes[sizes > 1].sum() / len(codes)),
        **{f"level_{level + 1}_codes_used": int(len(np.unique(codes[:, level]))) for level in range(codes.shape[1])},
    }
