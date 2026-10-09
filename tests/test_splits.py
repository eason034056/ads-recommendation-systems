import pandas as pd

from adsrec.splits import chronological_split


def test_chronological_split_is_ordered_and_disjoint():
    train, validation, test = chronological_split(pd.DataFrame({"time_ms": list(range(20))}), train_fraction=.5, validation_fraction=.25)
    assert train.time_ms.max() < validation.time_ms.min() < test.time_ms.min()
    assert len(train) + len(validation) + len(test) == 20
