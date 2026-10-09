import numpy as np
import pytest

from adsrec.evaluation import binary_metrics, ranking_metrics


def test_ranking_metrics_perfect_ranking():
    values = ranking_metrics(np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1]), (1,))
    assert values["recall@1"] == 1.0
    assert values["mrr"] == 1.0


def test_binary_metrics_calibration_compares_mean_prediction_to_rate():
    assert binary_metrics(np.array([0, 0, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))["calibration"] == 2.0


def test_binary_metrics_rejects_single_class():
    with pytest.raises(ValueError):
        binary_metrics(np.ones(3), np.array([0.2, 0.3, 0.4]))
