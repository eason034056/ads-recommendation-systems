import pandas as pd

from adsrec.datasets import make_ranking_data


def test_ranking_maps_unknown_test_categories_to_zero():
    frame = pd.DataFrame({"user_id": [1] * 7 + [2] * 3, "video_id": [1] * 7 + [2] * 3, "tab": [1] * 10, "time_ms": range(10), "is_click": [0, 1] * 5, "long_view": [0, 1] * 5})
    result = make_ranking_data(frame)
    assert (result.test[0][:, 0] == 0).all()
    assert (result.test[0][:, 1] == 0).all()
