import numpy as np

from adsrec.semantic_ids import collision_stats, cooccurrence_vectors, deduplicate


def test_cooccurrence_vectors_drop_padding_and_zero_isolated_items():
    vectors = cooccurrence_vectors(np.array([[1], [3]]), np.array([2, 4]), item_count=6, dimension=2, seed=0)
    assert vectors.shape == (5, 2)
    assert np.allclose(vectors[4], 0) and not np.allclose(vectors[:4], 0)


def test_deduplicate_makes_ids_unique_and_ranks_by_popularity():
    codes = np.array([[0, 1], [0, 1], [2, 3], [0, 1]])
    result = deduplicate(codes, np.array([5, 9, 1, 7]))
    assert len(np.unique(result, axis=0)) == len(codes)
    assert result[:, 2].tolist() == [2, 0, 0, 1]


def test_collision_stats_counts_shared_ids():
    stats = collision_stats(np.array([[0, 1], [0, 1], [2, 3]]))
    assert stats["distinct_ids"] == 2 and stats["largest_group"] == 2
    assert stats["items_sharing_an_id"] == 2 / 3
