from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import chronological_split


@dataclass
class RetrievalData:
    train: tuple[np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    item_count: int


@dataclass
class RankingData:
    train: tuple[np.ndarray, np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray, np.ndarray]
    cardinalities: list[int]


def load_kuairand_logs(root: Path, max_rows: int) -> pd.DataFrame:
    files = sorted(root.glob("**/log_standard_*_pure.csv"))
    if len(files) != 2:
        raise FileNotFoundError("Expected the two KuaiRand-Pure standard log CSV files")
    per_file = max_rows // len(files)
    columns = ["user_id", "video_id", "time_ms", "is_click", "long_view", "tab"]
    parts = [pd.read_csv(path, usecols=columns, nrows=per_file) for path in files]
    return pd.concat(parts, ignore_index=True).sort_values("time_ms", kind="stable").reset_index(drop=True)


def _fit_mapping(values: pd.Series) -> dict[int, int]:
    return {int(value): index + 1 for index, value in enumerate(values.drop_duplicates())}


def make_retrieval_data(frame: pd.DataFrame, history_length: int = 5) -> RetrievalData:
    train_raw, validation_raw, test_raw = chronological_split(frame)
    clicked_train = train_raw[train_raw.is_click == 1]
    user_map = _fit_mapping(clicked_train.user_id)
    item_map = _fit_mapping(clicked_train.video_id)
    partition = {index: "train" for index in train_raw.index}
    partition.update({index: "validation" for index in validation_raw.index})
    partition.update({index: "test" for index in test_raw.index})
    examples: dict[str, list[tuple[list[int], int]]] = {key: [] for key in ("train", "validation", "test")}
    history: dict[int, list[int]] = {}
    for index, row in frame[frame.is_click == 1].iterrows():
        user = user_map.get(int(row.user_id))
        item = item_map.get(int(row.video_id))
        if user is None or item is None:
            continue
        previous = history.setdefault(user, [])
        if len(previous) >= history_length:
            examples[partition[index]].append((previous[-history_length:], item))
        previous.append(item)
    def pack(key: str) -> tuple[np.ndarray, np.ndarray]:
        rows = examples[key]
        return np.asarray([row[0] for row in rows], dtype=np.int64), np.asarray([row[1] for row in rows], dtype=np.int64)
    return RetrievalData(pack("train"), pack("validation"), pack("test"), len(item_map) + 1)


def make_ranking_data(frame: pd.DataFrame) -> RankingData:
    train, validation, test = chronological_split(frame)
    mappings = [_fit_mapping(train[column]) for column in ("user_id", "video_id", "tab")]
    def pack(partition: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        copy = partition.copy()
        hour = (copy.time_ms // 3_600_000 % 24).astype(int) + 1
        feature_columns = [copy.user_id.map(mappings[0]).fillna(0), copy.video_id.map(mappings[1]).fillna(0), hour, copy.tab.map(mappings[2]).fillna(0)]
        features = np.stack([column.to_numpy(dtype=np.int64) for column in feature_columns], axis=1)
        click = copy.is_click.to_numpy(dtype=np.float32)
        post_click = (copy.is_click.astype(bool) & copy.long_view.astype(bool)).to_numpy(dtype=np.float32)
        return features, click, post_click
    return RankingData(pack(train), pack(validation), pack(test), [len(mapping) + 1 for mapping in mappings[:2]] + [25, len(mappings[2]) + 1])
