import pandas as pd

from adsrec.datasets import load_kuairand_logs, make_ranking_data


def test_ranking_maps_unknown_test_categories_to_zero():
    frame = pd.DataFrame({"user_id": [1] * 7 + [2] * 3, "video_id": [1] * 7 + [2] * 3, "tab": [1] * 10, "time_ms": range(10), "is_click": [0, 1] * 5, "long_view": [0, 1] * 5})
    result = make_ranking_data(frame)
    assert (result.test[0][:, 0] == 0).all()
    assert (result.test[0][:, 1] == 0).all()


def test_kuairand_loader_keeps_every_row_of_selected_users_in_time_order(tmp_path):
    header = "user_id,video_id,time_ms,is_click,long_view,tab\n"
    (tmp_path / "log_standard_a_pure.csv").write_text(header + "0,1,30,1,0,1\n0,2,10,0,0,1\n5,3,5,1,1,1\n")
    (tmp_path / "log_standard_b_pure.csv").write_text(header + "0,4,40,1,0,1\n9,5,1,1,0,1\n")
    frame = load_kuairand_logs(tmp_path, users=6)
    assert frame.user_id.tolist() == [5, 0, 0, 0]
    assert frame.time_ms.is_monotonic_increasing
