from __future__ import annotations

import pandas as pd


def chronological_split(frame: pd.DataFrame, timestamp: str = "time_ms", train_fraction: float = 0.70, validation_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make non-overlapping time partitions; ties are placed in the later partition."""
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("fractions must be in increasing range below one")
    ordered = frame.sort_values(timestamp, kind="stable")
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:validation_end].copy(), ordered.iloc[validation_end:].copy()
