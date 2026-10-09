from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import chronological_split

FEATURES = ["uid", "campaign", *[f"cat{i}" for i in range(1, 10)], "time_bucket"]


@dataclass
class CriteoData:
    train: tuple[np.ndarray, np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray, np.ndarray]
    cardinalities: list[int]


def load_criteo_sample(path: Path, rows: int) -> pd.DataFrame:
    fields = ["timestamp", "uid", "campaign", "click", "conversion", *[f"cat{i}" for i in range(1, 10)]]
    frame = pd.read_csv(path, sep="\t", compression="gzip", usecols=fields, nrows=rows)
    return frame.sort_values("timestamp", kind="stable")


def make_criteo_data(frame: pd.DataFrame, hash_buckets: int = 8192) -> CriteoData:
    train, validation, test = chronological_split(frame, "timestamp")
    def pack(partition: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        copy = partition.copy()
        copy["time_bucket"] = (copy.timestamp // 86_400).astype(np.int64)
        columns = [np.mod(copy[name].fillna(-1).astype(np.int64).to_numpy(), hash_buckets) + 1 for name in FEATURES]
        features = np.stack(columns, axis=1)
        clicks = copy.click.to_numpy(dtype=np.float32)
        click_conversion = (copy.click.astype(bool) & copy.conversion.astype(bool)).to_numpy(dtype=np.float32)
        return features, clicks, click_conversion
    return CriteoData(pack(train), pack(validation), pack(test), [hash_buckets + 1] * len(FEATURES))
