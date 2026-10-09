import numpy as np

from adsrec.linear import fit_logistic, one_hot


def test_one_hot_places_each_field_in_its_own_column_block():
    assert one_hot(np.array([[0, 2], [1, 0]]), [2, 3]).toarray().tolist() == [[1, 0, 0, 0, 1], [0, 1, 1, 0, 0]]


def test_fit_logistic_prefers_weak_regularization_on_a_separable_field():
    features = np.array([[0], [1]] * 50)
    labels = features[:, 0].astype(float)
    model, strength = fit_logistic((features, labels), (features, labels), [2])
    assert strength == 1.0
    assert (model.predict_proba(one_hot(features, [2]))[:, 1].round() == labels).all()
