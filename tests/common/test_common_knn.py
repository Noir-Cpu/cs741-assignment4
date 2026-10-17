"""Tests for common.knn against hand-computed examples, brute force and scikit-learn."""
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier as SKKNN

from common import knn, protocol


def brute_force(train, query, k, p):
    train = train.astype(np.float64)
    query = query.astype(np.float64)
    diff = np.abs(query[:, None, :] - train[None, :, :])
    dist = diff.sum(axis=2) if p == 1 else np.sqrt((diff ** 2).sum(axis=2))
    order = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(dist, order, axis=1), order


def test_hand_computed_one_dimensional_example():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    y = np.array([0, 0, 0, 1, 1])
    model = knn.KNNClassifier(3, "euclidean").fit(X, y)
    dist, idx = model.kneighbors(np.array([[4.0]]), 3)
    assert list(idx[0]) == [2, 1, 0] and np.allclose(dist[0], [2.0, 3.0, 4.0])
    assert model.predict(np.array([[4.0], [9.5]]))[0] == 0
    assert model.predict(np.array([[9.5]]))[0] == 1
    manhattan = knn.KNNClassifier(2, "manhattan").fit(np.array([[0.0, 0.0], [2.0, 0.5], [3.0, 0.0]]),
                                                       np.array([0, 1, 1]))
    dist, idx = manhattan.kneighbors(np.array([[2.5, 0.0]]), 2)
    assert list(idx[0]) == [2, 1] and np.allclose(dist[0], [0.5, 1.0])


def test_distance_weighting_changes_the_vote():
    X = np.array([[0.0], [3.0], [4.0]])
    y = np.array([1, 0, 0])
    q = np.array([[1.0]])        # distances 1, 2, 3
    assert knn.KNNClassifier(3, weights="uniform").fit(X, y).predict(q)[0] == 0
    weighted = knn.KNNClassifier(3, weights="distance").fit(X, y)
    assert weighted.predict(q)[0] == 1
    scores = weighted.decision_scores(q)[0]
    assert scores[1] == pytest.approx(1.0) and scores[0] == pytest.approx(0.5 + 1 / 3)
    assert weighted.predict_proba(q)[0].sum() == pytest.approx(1.0)


def test_exact_match_dominates_a_distance_weighted_vote():
    X = np.array([[0.0], [1.0], [1.5]])
    y = np.array([1, 0, 0])
    assert knn.KNNClassifier(3, weights="distance").fit(X, y).predict(np.array([[0.0]]))[0] == 1


def test_class_weight_vectors():
    y = np.array([1, 0, 0, 0])
    assert np.allclose(knn.class_weight_vector(y, 2, None), [1.0, 1.0])
    balanced = knn.class_weight_vector(y, 2, "balanced")
    assert np.allclose(balanced, [4 / (2 * 3), 4 / (2 * 1)])
    assert np.allclose(knn.class_weight_vector(y, 2, 0.5), np.sqrt(balanced))
    assert np.allclose(knn.class_weight_vector(y, 2, 1.0), balanced)
    assert knn.class_weight_vector(y, 3, "balanced")[2] == 0.0       # absent class
    with pytest.raises(ValueError):
        knn.class_weight_vector(y, 2, "other")


def test_class_weighted_vote_flips_a_majority():
    X = np.array([[0.0], [1.0], [2.0], [9.0]])
    y = np.array([1, 0, 0, 0])
    q = np.array([[1.0]])           # three nearest: classes 0, 1, 0 in order of distance 0, 1, 1
    assert knn.KNNClassifier(3).fit(X, y).predict(q)[0] == 0
    assert knn.KNNClassifier(3, class_weight="balanced").fit(X, y).predict(q)[0] == 1


@pytest.mark.parametrize("metric,p", [("euclidean", 2), ("manhattan", 1)])
@pytest.mark.parametrize("n_train,d", [(300, 5), (3100, 40)])      # 3100 rows span several tiles
def test_neighbours_match_brute_force(metric, p, n_train, d):
    gen = np.random.default_rng(protocol.seed_int("test-knn-brute"))
    train = gen.normal(size=(n_train, d)).astype(np.float32)
    query = gen.normal(size=(37, d)).astype(np.float32)
    y = gen.integers(0, 4, n_train)
    model = knn.KNNClassifier(9, metric).fit(train, y)
    dist, idx = model.kneighbors(query, 9)
    ref_dist, ref_idx = brute_force(train, query, 9, p)
    assert np.array_equal(idx, ref_idx)
    assert np.allclose(dist, ref_dist, rtol=1e-4)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_predictions_match_scikit_learn(metric, weights):
    gen = np.random.default_rng(protocol.seed_int("test-knn-sklearn"))
    train = gen.normal(size=(800, 6)).astype(np.float32)
    y = (train[:, 0] + 0.5 * gen.normal(size=800) > 0).astype(int) + (train[:, 1] > 1)
    query = gen.normal(size=(200, 6)).astype(np.float32)
    ours = knn.KNNClassifier(7, metric, weights).fit(train, y)
    ref = SKKNN(7, metric=metric, weights=weights).fit(train, y)
    assert np.array_equal(ours.predict(query), ref.predict(query))
    assert np.allclose(ours.predict_proba(query), ref.predict_proba(query), atol=1e-5)


def test_tie_in_distance_prefers_the_lower_training_index():
    X = np.array([[1.0], [1.0], [1.0], [5.0]])
    y = np.array([0, 1, 0, 1])
    _, idx = knn.KNNClassifier(2).fit(X, y).kneighbors(np.array([[1.0]]), 2)
    assert list(idx[0]) == [0, 1]


def test_vote_from_stored_neighbours_equals_direct_prediction():
    gen = np.random.default_rng(protocol.seed_int("test-knn-stored"))
    train = gen.normal(size=(500, 4)).astype(np.float32)
    y = gen.integers(0, 3, 500)
    query = gen.normal(size=(60, 4)).astype(np.float32)
    big = knn.KNNClassifier(15).fit(train, y)
    dist, idx = big.kneighbors(query, 15)
    for k in (1, 5, 15):
        direct = knn.KNNClassifier(k, weights="distance").fit(train, y).predict(query)
        stored = knn.KNNClassifier.predict_from_neighbours(idx, dist, big.y_, k, "distance",
                                                           np.ones(3), 3)
        assert np.array_equal(direct, stored)


def test_k_larger_than_training_set_is_rejected():
    model = knn.KNNClassifier(3).fit(np.zeros((2, 2)), np.array([0, 1]))
    with pytest.raises(ValueError):
        model.kneighbors(np.zeros((1, 2)), 3)
    with pytest.raises(ValueError):
        knn.KNNClassifier(3, metric="cosine")


def test_large_training_set_runs_in_small_memory():
    gen = np.random.default_rng(protocol.seed_int("test-knn-large"))
    train = gen.normal(size=(60000, 12)).astype(np.float32)
    y = gen.integers(0, 3, 60000)
    query = gen.normal(size=(40, 12)).astype(np.float32)
    dist, idx = knn.KNNClassifier(5).fit(train, y).kneighbors(query, 5)
    d0 = np.sqrt(((train.astype(np.float64) - query[0]) ** 2).sum(axis=1))
    assert np.array_equal(idx[0], np.argsort(d0, kind="stable")[:5])
