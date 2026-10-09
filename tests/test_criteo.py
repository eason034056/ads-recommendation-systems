import pandas as pd

from adsrec.criteo import make_criteo_data


def test_criteo_click_conversion_requires_a_click():
    frame = pd.DataFrame({"timestamp": range(12), "uid": range(12), "campaign": [1] * 12, "click": [0, 1] * 6, "conversion": [1] * 12, **{f"cat{i}": [1] * 12 for i in range(1, 10)}})
    data = make_criteo_data(frame, hash_buckets=10)
    assert data.train[2].sum() == data.train[1].sum()
