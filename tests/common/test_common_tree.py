"""Tests for common.tree against hand-computed examples and scikit-learn."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier as SKTree

from common import protocol, tree


def make_data(n=600, p=5, n_classes=3, tag="test-tree-data"):
    gen = np.random.default_rng(protocol.seed_int(tag))
    X = gen.normal(size=(n, p))
    score = X[:, 0] + X[:, 1] ** 2 + 0.4 * gen.normal(size=n)
    y = (score > 0.4).astype(int) + (X[:, 2] > 1.0)
    return X, np.minimum(y, n_classes - 1)


def test_stump_threshold_on_a_separable_toy_problem():
    X = np.arange(1.0, 7.0).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = tree.DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert model.n_nodes_ == 3 and model.feature_[0] == 0
    assert model.threshold_[0] == pytest.approx(3.5)
    assert list(model.predict(np.array([[0.0], [3.5], [3.6], [100.0]]))) == [0, 0, 1, 1]
    assert np.allclose(model.predict_proba(np.array([[1.0]])), [[1.0, 0.0]])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_best_split_of_a_hand_computed_example(criterion):
    # y = 0, 0, 1, 0. Splitting after the second value gives children {0, 0} and {1, 0}.
    # Gini cost: 0 + 2 * 0.5 = 1.0 against 4 * 0.375 = 1.5 for the parent, so the decrease is 0.5.
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 0])
    model = tree.DecisionTreeClassifier(criterion, max_depth=1).fit(X, y)
    assert model.threshold_[0] == pytest.approx(2.5)
    if criterion == "gini":
        assert tree._parent_cost(np.array([3.0, 1.0]), 4.0, 0) == pytest.approx(1.5)
        assert tree._split_cost(np.array([2.0, 0.0]), np.array([3.0, 1.0]), 4.0, 0) == pytest.approx(1.0)
    else:
        parent = tree._parent_cost(np.array([3.0, 1.0]), 4.0, 1)
        assert parent == pytest.approx(-(3 * np.log(0.75) + 1 * np.log(0.25)))
        split = tree._split_cost(np.array([2.0, 0.0]), np.array([3.0, 1.0]), 4.0, 1)
        assert split == pytest.approx(-(1 * np.log(0.5) + 1 * np.log(0.5)))
    assert model.feature_importances_[0] == pytest.approx(1.0)


def test_min_samples_leaf_restricts_the_split():
    X = np.arange(1.0, 7.0).reshape(-1, 1)
    y = np.array([0, 1, 1, 1, 1, 1])
    assert tree.DecisionTreeClassifier(max_depth=1).fit(X, y).threshold_[0] == pytest.approx(1.5)
    constrained = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=2).fit(X, y)
    assert constrained.threshold_[0] == pytest.approx(2.5)
    full = tree.DecisionTreeClassifier(min_samples_leaf=3).fit(X, y)
    leaves = full.children_left_ < 0
    assert (full.n_node_samples_[leaves] >= 3).all()


def test_depth_and_leaf_constraints_hold_on_random_data():
    X, y = make_data(1500)
    for depth in (1, 3, 6):
        assert tree.DecisionTreeClassifier(max_depth=depth).fit(X, y).depth_ <= depth
    model = tree.DecisionTreeClassifier(min_samples_leaf=17).fit(X, y)
    assert (model.n_node_samples_[model.children_left_ < 0] >= 17).all()
    assert tree.DecisionTreeClassifier(min_samples_split=400).fit(X, y).n_nodes_ < 30


def test_unlimited_tree_fits_training_data_without_duplicates():
    X, y = make_data(500)
    assert (tree.DecisionTreeClassifier().fit(X, y).predict(X) == y).all()


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_stump_matches_scikit_learn(criterion, class_weight):
    X, y = make_data(700)
    ours = tree.DecisionTreeClassifier(criterion, max_depth=1, class_weight=class_weight,
                                       max_bins=1000).fit(X, y)
    ref = SKTree(criterion=criterion, max_depth=1, class_weight=class_weight).fit(X, y)
    assert ours.feature_[0] == ref.tree_.feature[0]
    assert ours.threshold_[0] == pytest.approx(ref.tree_.threshold[0])
    assert np.array_equal(ours.predict(X), ref.predict(X))


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_depth_limited_tree_agrees_with_scikit_learn(criterion, class_weight):
    X, y = make_data(1500)
    Xtr, ytr, Xte = X[:1000], y[:1000], X[1000:]
    ours = tree.DecisionTreeClassifier(criterion, max_depth=7, min_samples_leaf=3,
                                       class_weight=class_weight, max_bins=2000).fit(Xtr, ytr)
    ref = SKTree(criterion=criterion, max_depth=7, min_samples_leaf=3, class_weight=class_weight,
                 random_state=protocol.seed_int("test-tree-sk")).fit(Xtr, ytr)
    assert (ours.predict(Xtr) == ref.predict(Xtr)).mean() > 0.97
    assert (ours.predict(Xte) == ref.predict(Xte)).mean() > 0.93
    assert abs(ours.n_nodes_ - ref.tree_.node_count) <= 0.1 * ref.tree_.node_count + 2


def test_sample_weight_equals_duplicated_rows():
    X, y = make_data(300)
    counts = np.random.default_rng(protocol.seed_int("test-tree-weights")).integers(0, 4, 300)
    weighted = tree.DecisionTreeClassifier(max_depth=5, max_bins=500).fit(X, y, sample_weight=counts,
                                                                           n_classes=3)
    repeated = np.repeat(np.arange(300), counts)
    duplicated = tree.DecisionTreeClassifier(max_depth=5, max_bins=500).fit(X[repeated], y[repeated],
                                                                             n_classes=3)
    grid = np.random.default_rng(protocol.seed_int("test-tree-grid")).normal(size=(500, 5))
    assert (weighted.predict(grid) == duplicated.predict(grid)).mean() > 0.98
    assert weighted.value_[0].sum() == pytest.approx(counts.sum())


def test_zero_weight_rows_are_ignored():
    X, y = make_data(400)
    w = np.zeros(400)
    w[:200] = 1.0
    a = tree.DecisionTreeClassifier(max_depth=4, max_bins=500).fit(X, y, sample_weight=w, n_classes=3)
    b = tree.DecisionTreeClassifier(max_depth=4, max_bins=500).fit(X[:200], y[:200], n_classes=3)
    assert a.value_[0].sum() == pytest.approx(200.0)
    assert np.array_equal(a.feature_, b.feature_)
    with pytest.raises(ValueError):
        tree.DecisionTreeClassifier().fit(X, y, sample_weight=np.zeros(400))


def test_class_weight_forms():
    y = np.array([0, 0, 0, 1])
    sw = np.ones(4)
    assert np.allclose(tree.class_weight_array("balanced", y, sw, 2), [4 / 6, 2.0])
    assert np.allclose(tree.class_weight_array({1: 5.0}, y, sw, 2), [1.0, 5.0])
    assert np.allclose(tree.class_weight_array(None, y, sw, 2), [1.0, 1.0])
    assert np.allclose(tree.class_weight_array([2.0, 3.0], y, sw, 2), [2.0, 3.0])
    assert tree.class_weight_array("balanced", y, sw, 3)[2] == 0.0
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    # 3 versus 1 with a tie-breaking boundary row: balanced weights lift the minority class
    model = tree.DecisionTreeClassifier(max_depth=0, class_weight="balanced").fit(X, y)
    assert model.value_[0].tolist() == pytest.approx([2.0, 2.0])


def test_max_features_resolution():
    assert tree.resolve_max_features(None, 50) == 50
    assert tree.resolve_max_features("sqrt", 50) == 7
    assert tree.resolve_max_features("log2", 64) == 6
    assert tree.resolve_max_features(0.2, 50) == 10
    assert tree.resolve_max_features(3, 50) == 3
    assert tree.resolve_max_features(500, 50) == 50
    with pytest.raises(ValueError):
        tree.resolve_max_features("all", 5)


def test_feature_subsets_are_deterministic_given_the_seed():
    X, y = make_data(800, p=12)
    a = tree.DecisionTreeClassifier(max_features=3, seed=11).fit(X, y)
    b = tree.DecisionTreeClassifier(max_features=3, seed=11).fit(X, y)
    c = tree.DecisionTreeClassifier(max_features=3, seed=12).fit(X, y)
    assert np.array_equal(a.feature_, b.feature_) and np.array_equal(a.threshold_, b.threshold_)
    assert not (a.feature_.shape == c.feature_.shape and np.array_equal(a.feature_, c.feature_))
    full_a = tree.DecisionTreeClassifier(seed=11).fit(X, y)
    full_b = tree.DecisionTreeClassifier(seed=99).fit(X, y)
    assert np.array_equal(full_a.feature_, full_b.feature_)         # seed has no effect


def test_feature_subsets_restrict_every_split_and_still_skip_constant_columns():
    X, y = make_data(600, p=6)
    X[:, 0] = 1.0                                   # constant column is never a split
    model = tree.DecisionTreeClassifier(max_features=1, seed=5).fit(X, y)
    assert 0 not in set(model.feature_[model.feature_ >= 0])
    assert (model.predict(X) == y).mean() > 0.9


def test_sort_path_and_histogram_path_build_identical_trees(monkeypatch):
    X, y = make_data(1200, p=6)
    X[:, 5] = np.round(X[:, 5], 1)                  # a feature with few distinct values
    monkeypatch.setattr(tree, "_SORT_RATIO", 0.0)   # histogram everywhere
    hist = tree.DecisionTreeClassifier(max_depth=9, class_weight="balanced").fit(X, y)
    monkeypatch.setattr(tree, "_SORT_RATIO", 1e9)   # sorting everywhere
    srt = tree.DecisionTreeClassifier(max_depth=9, class_weight="balanced").fit(X, y)
    assert np.array_equal(hist.feature_, srt.feature_)
    assert np.allclose(hist.threshold_, srt.threshold_)
    assert np.allclose(hist.value_, srt.value_)


def test_bin_mapper_exact_and_quantile_modes():
    values = np.array([3.0, 1.0, 2.0, 2.0, 5.0]).reshape(-1, 1)
    mapper = tree.BinMapper(10).fit(values)
    assert np.allclose(mapper.edges_[0], [1.5, 2.5, 4.0])
    assert list(mapper.transform(values)[0]) == [2, 0, 1, 1, 3]
    assert list(mapper.transform(np.array([[1.5], [1.5000001], [9.0]]))[0]) == [0, 1, 3]
    big = np.random.default_rng(protocol.seed_int("test-bins")).normal(size=(5000, 1))
    q = tree.BinMapper(16).fit(big)
    assert q.n_bins_[0] <= 16
    counts = np.bincount(q.transform(big)[0])
    assert counts.min() > 5000 / 16 * 0.5
    const = tree.BinMapper(8).fit(np.ones((10, 1)))
    assert const.n_bins_[0] == 1


def test_quantised_features_still_give_an_accurate_tree():
    X, y = make_data(3000)
    coarse = tree.DecisionTreeClassifier(max_depth=8, max_bins=16).fit(X[:2000], y[:2000])
    fine = tree.DecisionTreeClassifier(max_depth=8, max_bins=2000).fit(X[:2000], y[:2000])
    acc_coarse = (coarse.predict(X[2000:]) == y[2000:]).mean()
    acc_fine = (fine.predict(X[2000:]) == y[2000:]).mean()
    assert acc_coarse > acc_fine - 0.05 and acc_coarse > 0.75


def test_prebinned_data_can_be_shared_between_trees():
    X, y = make_data(800)
    mapper = tree.BinMapper().fit(X)
    Xb = mapper.transform(X)
    a = tree.DecisionTreeClassifier(max_depth=6).fit(X, y)
    b = tree.DecisionTreeClassifier(max_depth=6).fit(X, y, binner=mapper, Xb_t=Xb)
    assert np.array_equal(a.feature_, b.feature_) and np.allclose(a.threshold_, b.threshold_)


def test_trees_fit_in_parallel_threads_match_serial_fits():
    X, y = make_data(1500, p=8)
    mapper = tree.BinMapper(255).fit(X)
    Xb = mapper.transform(X)
    gen = np.random.default_rng(protocol.seed_int("test-tree-bootstrap"))
    weights = [np.bincount(gen.integers(0, 1500, 1500), minlength=1500).astype(float) for _ in range(8)]

    def fit(i):
        return tree.DecisionTreeClassifier(max_features="sqrt", seed=100 + i, max_depth=10).fit(
            X, y, sample_weight=weights[i], n_classes=3, binner=mapper, Xb_t=Xb)

    serial = [fit(i) for i in range(8)]
    with ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(fit, range(8)))
    for a, b in zip(serial, threaded):
        assert np.array_equal(a.feature_, b.feature_) and np.allclose(a.value_, b.value_)


def test_predict_probabilities_and_float32_input():
    X, y = make_data(500)
    model = tree.DecisionTreeClassifier(max_depth=4).fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0) and proba.shape == (500, 3)
    assert np.array_equal(model.predict(X), np.argmax(proba, axis=1))
    assert (model.predict(X.astype(np.float32)) == model.predict(X)).mean() > 0.99
    assert np.array_equal(model.predict(X, parallel=True), model.predict(X))
    assert model.feature_importances_.sum() == pytest.approx(1.0)


def test_invalid_arguments():
    X, y = make_data(50)
    with pytest.raises(ValueError):
        tree.DecisionTreeClassifier().fit(X, y, Xb_t=tree.BinMapper().fit_transform(X))
    with pytest.raises(ValueError):
        tree.DecisionTreeClassifier(criterion="log_loss")
    with pytest.raises(ValueError):
        tree.BinMapper(1)
    with pytest.raises(ValueError):
        tree.class_weight_array("auto", np.array([0, 1]), np.ones(2), 2)
