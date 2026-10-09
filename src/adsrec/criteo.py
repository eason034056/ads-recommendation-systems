from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import chronological_split

CATEGORICAL = ["uid", "campaign", *[f"cat{i}" for i in range(1, 10)]]
RECENCY_BUCKETS = 32


@dataclass
class CriteoData:
    train: tuple[np.ndarray, np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray, np.ndarray]
    cardinalities: list[int]
    feature_names: list[str]


def load_criteo_sample(path: Path, rows: int) -> pd.DataFrame:
    fields = ["timestamp", "uid", "campaign", "click", "conversion", "time_since_last_click", *CATEGORICAL[2:]]
    frame = pd.read_csv(path, sep="\t", compression="gzip", usecols=fields, nrows=rows)
    return frame.sort_values("timestamp", kind="stable")


def fit_vocabulary(values: pd.Series, min_count: int) -> dict[int, int]:
    """Index values seen at least `min_count` times; rarer and unseen values share index 0."""
    counts = values.value_counts()
    return {int(value): index + 1 for index, value in enumerate(counts.index[counts >= min_count])}


def recency_bucket(seconds: pd.Series) -> np.ndarray:
    """Log2 buckets of the time since the user's previous click; 0 means no earlier click on record."""
    values = seconds.to_numpy(dtype=np.float64)
    buckets = np.floor(np.log2(np.maximum(values, 0) + 1)).astype(np.int64) + 1
    return np.where(values < 0, 0, np.minimum(buckets, RECENCY_BUCKETS - 1))


def make_criteo_data(frame: pd.DataFrame, min_count: int = 2, use_recency: bool = False) -> CriteoData:
    train, validation, test = chronological_split(frame, "timestamp")
    # Vocabularies see only the training partition, so a value that first appears later maps to index 0.
    # With min_count > 1, rare training values also map there, which trains row 0 as a "new value" embedding.
    vocabularies = {name: fit_vocabulary(train[name], min_count) for name in CATEGORICAL}
    def pack(partition: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        columns = [partition[name].map(vocabularies[name]).fillna(0).to_numpy(dtype=np.int64) for name in CATEGORICAL]
        if use_recency:
            columns.append(recency_bucket(partition.time_since_last_click))
        clicks = partition.click.to_numpy(dtype=np.float32)
        click_conversion = (partition.click.astype(bool) & partition.conversion.astype(bool)).to_numpy(dtype=np.float32)
        return np.stack(columns, axis=1), clicks, click_conversion
    cardinalities = [len(vocabularies[name]) + 1 for name in CATEGORICAL] + ([RECENCY_BUCKETS] if use_recency else [])
    feature_names = CATEGORICAL + (["recency"] if use_recency else [])
    return CriteoData(pack(train), pack(validation), pack(test), cardinalities, feature_names)
