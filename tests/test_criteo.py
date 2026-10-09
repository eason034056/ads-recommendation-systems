import pandas as pd

from adsrec.criteo import make_criteo_data, recency_bucket


def criteo_frame(rows: int, **overrides) -> pd.DataFrame:
    columns = {"timestamp": range(rows), "uid": range(rows), "campaign": [1] * rows, "click": [0, 1] * (rows // 2), "conversion": [1] * rows,
               "time_since_last_click": [-1] * rows, **{f"cat{i}": [1] * rows for i in range(1, 10)}}
    return pd.DataFrame({**columns, **overrides})


def test_criteo_click_conversion_requires_a_click():
    data = make_criteo_data(criteo_frame(12))
    assert data.train[2].sum() == data.train[1].sum()


def test_criteo_vocabulary_ignores_rare_and_future_values():
    # Rows 0-6 are training: campaign 5 is frequent, 6 is a singleton, and 7 only appears afterwards.
    data = make_criteo_data(criteo_frame(10, campaign=[5, 5, 6, 5, 5, 5, 5, 7, 7, 7]), min_count=2)
    campaign = data.feature_names.index("campaign")
    assert data.train[0][:, campaign].tolist() == [1, 1, 0, 1, 1, 1, 1]
    assert (data.test[0][:, campaign] == 0).all()


def test_recency_bucket_separates_missing_and_caps_large_gaps():
    assert recency_bucket(pd.Series([-1, 0, 1, 3, 10 ** 12])).tolist() == [0, 1, 2, 3, 31]
