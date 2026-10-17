"""Tests for the ensembles: hand-computed toy examples and scikit-learn as oracle."""
import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier as SKAda
from sklearn.ensemble import RandomForestClassifier as SKForest
from sklearn.tree import DecisionTreeClassifier as SKTree

from common import protocol
from common.tree import DecisionTreeClassifier
from ensemble import diversity, ensemble, features


def make_data(n=900, p=6, tag="test-opt1-data"):
    gen = np.random.default_rng(protocol.seed_int(tag))
    X = gen.normal(size=(n, p))
    score = X[:, 0] + X[:, 1] ** 2 + 0.5 * gen.normal(size=n)
    y = (score > 0.5).astype(int) + (X[:, 2] > 1.0)
    return X, np.minimum(y, 2)


# ---------------------------------------------------------------------------
# class weights and sampling
# ---------------------------------------------------------------------------

def test_class_weight_vector_hand_computed():
    y = np.array([0, 0, 0, 0, 1, 1, 2])           # n = 7, m = 3, counts 4, 2, 1
    b = ensemble.class_weight_vector(y, 3, 1.0)
    assert np.allclose(b, [7 / 12, 7 / 6, 7 / 3])
    assert np.allclose(ensemble.class_weight_vector(y, 3, 0.0), 1.0)
    assert np.allclose(ensemble.class_weight_vector(y, 3, 0.5), np.sqrt([7 / 12, 7 / 6, 7 / 3]))
    # an absent class gets weight zero and does not count in m
    assert np.allclose(ensemble.class_weight_vector(np.array([0, 0, 1]), 3, 1.0), [0.75, 1.5, 0.0])


def test_standard_bootstrap_has_n_draws_and_about_63_percent_distinct_rows():
    y = np.zeros(5000, dtype=int)
    counts = np.array([5000.0])
    w = ensemble.draw_multiplicities(y, counts, "bootstrap", 0.0, None, protocol.rng("test-opt1/boot"))
    assert w.sum() == 5000
    assert (w > 0).mean() == pytest.approx(1 - np.exp(-1), abs=0.02)


def test_balanced_bootstrap_gives_equal_expected_class_counts():
    y = np.repeat([0, 1, 2], [3000, 600, 60])
    counts = np.bincount(y).astype(float)
    total = np.zeros(3)
    gen = protocol.rng("test-opt1/balanced")
    for _ in range(40):
        w = ensemble.draw_multiplicities(y, counts, "bootstrap", 1.0, None, gen)
        assert w.sum() == y.size
        total += np.bincount(y, weights=w)
    assert np.allclose(total / 40 / (y.size / 3), 1.0, atol=0.05)


def test_undersampling_draws_the_smallest_class_size_from_every_class():
    y = np.repeat([0, 1, 2], [300, 60, 20])
    counts = np.bincount(y).astype(float)
    w = ensemble.draw_multiplicities(y, counts, "under", 0.0, None, protocol.rng("test-opt1/under"))
    assert np.allclose(np.bincount(y, weights=w), 20)
    w2 = ensemble.draw_multiplicities(y, counts, "under", 0.0, 50, protocol.rng("test-opt1/under"))
    assert np.allclose(np.bincount(y, weights=w2), 50)


# ---------------------------------------------------------------------------
# random forest
# ---------------------------------------------------------------------------

def test_forest_is_deterministic_and_independent_of_the_number_of_threads():
    X, y = make_data()
    a = ensemble.RandomForestClassifier(20, n_jobs=1, purpose="test-opt1/det").fit(X, y, n_classes=3)
    b = ensemble.RandomForestClassifier(20, n_jobs=4, purpose="test-opt1/det").fit(X, y, n_classes=3)
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))
    c = ensemble.RandomForestClassifier(20, n_jobs=4, purpose="test-opt1/other").fit(X, y, n_classes=3)
    assert not np.array_equal(a.predict_proba(X), c.predict_proba(X))


def test_forest_probes_checkpoints_and_members_match_predict():
    X, y = make_data()
    Xt, yt = make_data(300, tag="test-opt1-data-b")
    rf = ensemble.RandomForestClassifier(12, purpose="test-opt1/probe", n_jobs=3)
    rf.fit(X, y, n_classes=3, probes={"t": Xt}, checkpoints=[1, 5])
    assert np.array_equal(rf.probe_pred_["t"][12], rf.predict(Xt))
    assert np.allclose(rf.probe_proba_["t"], rf.predict_proba(Xt), atol=1e-6)
    assert rf.probe_members_["t"].shape == (300, 12)
    assert np.array_equal(rf.probe_members_["t"][:, 0], rf.trees_[0].predict(Xt))
    # the first checkpoint is the first tree alone
    assert np.array_equal(rf.probe_pred_["t"][1], rf.trees_[0].predict(Xt))
    five = sum(t.predict_proba(Xt) for t in rf.trees_[:5])
    assert np.array_equal(rf.probe_pred_["t"][5], np.argmax(five, axis=1))


def test_out_of_bag_rows_are_the_rows_with_zero_multiplicity():
    X, y = make_data(400)
    rf = ensemble.RandomForestClassifier(30, purpose="test-opt1/oob", n_jobs=2)
    rf.fit(X, y, n_classes=3, track_oob=True, checkpoints=[30])
    # a row is out of bag for about 36.8 percent of the trees
    assert rf.oob_votes_.mean() / 30 == pytest.approx(np.exp(-1), abs=0.03)
    assert rf.oob_fraction_.mean() == pytest.approx(np.exp(-1), abs=0.03)
    # recompute the out-of-bag distribution of row 0 by hand from the stored trees and samples
    total = np.zeros(3)
    votes = 0
    for i, tree in enumerate(rf.trees_):
        gen = protocol.rng(f"test-opt1/oob/tree{i}/sample")
        w = ensemble.draw_multiplicities(y, np.bincount(y).astype(float), "bootstrap", 0.0, None, gen)
        if w[0] == 0:
            total += tree.predict_proba(X[:1])[0]
            votes += 1
    assert votes == rf.oob_votes_[0]
    assert np.allclose(rf.oob_proba_[0], total, atol=1e-5)
    # the out-of-bag error estimate is close to the error on fresh data
    Xt, yt = make_data(3000, tag="test-opt1-data-c")
    pred = rf.oob_pred_[30]
    covered = pred != 255
    assert abs((pred[covered] != y[covered]).mean() - (rf.predict(Xt) != yt).mean()) < 0.06


def test_forest_accuracy_is_close_to_scikit_learn():
    X, y = make_data(1500)
    Xt, yt = make_data(2000, tag="test-opt1-data-d")
    ours = ensemble.RandomForestClassifier(60, max_features="sqrt", purpose="test-opt1/sk", n_jobs=3)
    ours.fit(X, y, n_classes=3)
    ref = SKForest(60, max_features="sqrt", n_jobs=3, random_state=protocol.seed_int("test-opt1/skref")).fit(X, y)
    acc_ours = (ours.predict(Xt) == yt).mean()
    acc_ref = (ref.predict(Xt) == yt).mean()
    assert abs(acc_ours - acc_ref) < 0.03
    single = DecisionTreeClassifier(seed=protocol.seed_int("test-opt1/single")).fit(X, y, n_classes=3)
    assert acc_ours > (single.predict(Xt) == yt).mean()


def test_class_weights_raise_recall_of_the_rare_class():
    gen = protocol.rng("test-opt1/imbalance")
    n = 3000
    y = (gen.random(n) < 0.05).astype(int)
    X = gen.normal(size=(n, 4)) + 0.9 * y[:, None]
    plain = ensemble.RandomForestClassifier(40, max_depth=6, purpose="test-opt1/w", n_jobs=2).fit(X, y, n_classes=2)
    weighted = ensemble.RandomForestClassifier(40, max_depth=6, class_weight_power=1.0,
                                               purpose="test-opt1/w", n_jobs=2).fit(X, y, n_classes=2)
    rec = lambda m: (m.predict(X)[y == 1] == 1).mean()
    assert rec(weighted) > rec(plain)


# ---------------------------------------------------------------------------
# adaptive boosting
# ---------------------------------------------------------------------------

def test_samme_hand_computed_first_round():
    # Four instances, two classes, a depth-one tree. Feature values 1, 2, 3, 4, labels 0, 0, 1, 0.
    # The first stump splits at 2.5 (children {0, 0} and {1, 0}); the tied right child predicts the
    # lowest label 0, hence only instance 3 is wrong (weight 1/4). With m = 2 the model weight is
    # ln(3) + ln(1) = ln 3 and the wrong instance receives 0.25 * 3 = 0.75 before normalisation,
    # the others 0.25 each: new weights 0.25 / 1.5 = 1/6 for three instances and 0.75 / 1.5 = 1/2.
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 0])
    ab = ensemble.AdaBoostClassifier(1, max_depth=1, purpose="test-opt1/ada1").fit(X, y, n_classes=2, keep_trees=True)
    assert ab.errors_[0] == pytest.approx(0.25)
    assert ab.alphas_[0] == pytest.approx(np.log(3.0))
    w = np.full(4, 0.25)
    w[2] *= np.exp(ab.alphas_[0])
    w /= w.sum()
    assert np.allclose(w, [1 / 6, 1 / 6, 1 / 2, 1 / 6])
    assert list(ab.predict(X)) == [0, 0, 0, 0]


def test_samme_matches_scikit_learn_on_stumps():
    X, y = make_data(1200)
    Xt, yt = make_data(2000, tag="test-opt1-data-e")
    ours = ensemble.AdaBoostClassifier(40, max_depth=1, purpose="test-opt1/ada2").fit(X, y, n_classes=3, keep_trees=True)
    ref = SKAda(SKTree(max_depth=1), n_estimators=40,
                random_state=protocol.seed_int("test-opt1/skada")).fit(X, y)
    assert abs((ours.predict(Xt) == yt).mean() - (ref.predict(Xt) == yt).mean()) < 0.03
    # the weighted errors of the first rounds equal those of scikit-learn
    assert np.allclose(ours.errors_[:5], ref.estimator_errors_[:5], atol=1e-3)
    assert ours.alphas_[0] == pytest.approx(ref.estimator_weights_[0], rel=0.01)


def test_boosting_probe_checkpoints_agree_with_a_refit_prefix():
    X, y = make_data(600)
    Xt, _ = make_data(200, tag="test-opt1-data-f")
    full = ensemble.AdaBoostClassifier(10, max_depth=2, purpose="test-opt1/ada3")
    full.fit(X, y, n_classes=3, probes={"t": Xt}, checkpoints=[4], keep_trees=True)
    short = ensemble.AdaBoostClassifier(4, max_depth=2, purpose="test-opt1/ada3")
    short.fit(X, y, n_classes=3, probes={"t": Xt}, keep_trees=True)
    assert np.array_equal(full.probe_pred_["t"][4], short.probe_pred_["t"][4])
    assert np.array_equal(full.probe_pred_["t"][10], full.predict(Xt))


def test_boosting_reduces_the_training_error_of_stumps():
    X, y = make_data(800)
    ab = ensemble.AdaBoostClassifier(30, max_depth=2, purpose="test-opt1/ada4")
    ab.fit(X, y, n_classes=3, checkpoints=[1, 30])
    assert ab.train_error_[30] < ab.train_error_[1]


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------

def test_disagreement_hand_computed():
    # three members, two rows. Row 0: labels (0, 0, 1) has 1 agreeing pair of 3, so disagreement 2/3.
    # Row 1: labels (2, 2, 2) has 3 agreeing pairs, disagreement 0. Mean 1/3.
    members = np.array([[0, 0, 1], [2, 2, 2]], dtype=np.uint8)
    assert diversity.mean_disagreement(members, 3) == pytest.approx(1 / 3)
    # brute force over pairs on random labels
    gen = protocol.rng("test-opt1/dis")
    m = gen.integers(0, 4, size=(50, 7)).astype(np.uint8)
    brute = np.mean([(m[:, i] != m[:, j]).mean() for i in range(7) for j in range(i + 1, 7)])
    assert diversity.mean_disagreement(m, 4) == pytest.approx(brute)


def test_q_statistic_and_correlation_hand_computed():
    y = np.array([0, 0, 0, 0])
    # member A correct on rows 0, 1 and member B correct on rows 0, 2: N11 = 1, N00 = 1, N10 = 1, N01 = 1
    members = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert diversity.mean_q_statistic(members, y) == pytest.approx(0.0)
    assert diversity.error_correlation(members, y) == pytest.approx(0.0)
    # identical members give Q = 1 and correlation 1
    same = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    assert diversity.mean_q_statistic(same, y) == pytest.approx(1.0)
    assert diversity.error_correlation(same, y) == pytest.approx(1.0)
    assert diversity.member_accuracy(members, y) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_ordinal_encoding_is_frequency_rank_and_fitted_on_training_rows():
    from common import data
    table = data.load_table()
    train = table.take(np.arange(0, 20000))
    test = table.take(np.arange(20000, 25000))
    oh = features.ForestFeatures("onehot").fit(train)
    od = features.ForestFeatures("ordinal").fit(train)
    Xo, Xd = oh.transform(test), od.transform(test)
    assert Xd.shape[1] == 39 + 3 and Xo.shape[1] > Xd.shape[1]
    # the state codes ranked by training frequency: code 0 is the most frequent training state
    state_train = train.nominal[:, 2]
    most = np.bincount(state_train).argmax()
    assert np.all(Xd[test.nominal[:, 2] == most, -1] == 0)
    # the numeric block is untouched
    assert np.allclose(Xd[:, :39], test.numeric)
