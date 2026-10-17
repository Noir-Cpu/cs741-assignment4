"""Tests for common.data and common.preprocess."""
import numpy as np
import pytest

from common import data, preprocess

DATA_AVAILABLE = data.DATA_PATH.exists()


def toy_table(proto_codes, extra_numeric=None):
    names = ("dur", "sloss", "swin")
    numeric = np.array([[0.0, 5.0, 10.0], [np.e - 1, 6.0, 20.0], [np.e ** 2 - 1, 7.0, 30.0],
                        [0.0, 5.0, 40.0]])
    if extra_numeric is not None:
        numeric = extra_numeric
    proto_vocab = np.array(["arp", "tcp", "udp", "zzz"])
    n = numeric.shape[0]
    nominal = np.stack([np.asarray(proto_codes, dtype=np.int32),
                        np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)], axis=1)
    return data.Table(numeric=numeric.astype(np.float64), numeric_names=names, nominal=nominal,
                      nominal_names=("proto", "service", "state"),
                      vocab=(proto_vocab, np.array(["none"]), np.array(["INT"])),
                      y=np.arange(n) % 2, ids=np.arange(n))


def test_distance_preset_hand_computed():
    table = toy_table([1, 1, 2, 1])
    pre = preprocess.Preprocessor.for_knn(rare_threshold=2).fit(table)
    assert pre.numeric_names_ == ("dur", "swin")            # sloss dropped
    out = pre.transform(table)
    dur = np.log1p(table.numeric[:, 0])                      # 0, 1, 2, 0
    swin = table.numeric[:, 2]
    assert np.allclose(out[:, 0], (dur - dur.mean()) / dur.std())
    assert np.allclose(out[:, 1], (swin - swin.mean()) / swin.std())
    assert abs(out[:, 0].mean()) < 1e-12 and out[:, 0].std() == pytest.approx(1.0)
    # proto: tcp (3 rows) kept, udp (1 row) is rare and becomes "other"
    assert pre.categories_[0] == ("tcp", "other")
    assert np.array_equal(out[:, 2:4], [[1, 0], [1, 0], [0, 1], [1, 0]])


def test_statistics_come_from_training_rows_only():
    train = toy_table([1, 1, 1, 1])
    test = toy_table([1, 1, 1, 1], extra_numeric=np.array([[0.0, 5.0, 100.0]] * 2 + [[0.0, 5.0, 100.0]] * 2))
    pre = preprocess.Preprocessor.for_knn().fit(train)
    out = pre.transform(test)
    mean, std = train.numeric[:, 2].mean(), train.numeric[:, 2].std()
    assert np.allclose(out[:, 1], (100.0 - mean) / std)


def test_unseen_and_rare_protocols_map_to_other_and_unseen_elsewhere_is_zero():
    train = toy_table([1, 1, 1, 2])
    pre = preprocess.Preprocessor.for_tree(rare_threshold=2).fit(train)
    assert pre.categories_[0] == ("tcp", "other")
    test = toy_table([3, 0, 1, 2])                            # zzz and arp unseen in training
    out = pre.transform(test)
    first = pre.numeric_names_.__len__()
    assert np.array_equal(out[:, first:first + 2], [[0, 1], [0, 1], [1, 0], [0, 1]])
    plain = preprocess.Preprocessor(group_rare=False).fit(train)
    assert plain.categories_[0] == ("tcp", "udp")
    out = plain.transform(test)
    assert np.array_equal(out[0, first:first + 2], [0, 0])    # unseen: all zeros


def test_tree_preset_keeps_numeric_values_unchanged():
    table = toy_table([1, 1, 2, 1])
    pre = preprocess.Preprocessor.for_tree().fit(table)
    out = pre.transform(table)
    assert np.allclose(out[:, :3], table.numeric)
    assert pre.numeric_names_ == ("dur", "sloss", "swin")
    assert pre.feature_kinds_[0] == "numeric" and pre.feature_kinds_[-1].startswith("onehot:state")
    assert len(pre.feature_names_) == pre.n_features_ == out.shape[1]


def test_log_transform_rejects_negative_values():
    table = toy_table([1, 1, 1, 1], extra_numeric=np.array([[-1.0, 5.0, 1.0]] * 4))
    with pytest.raises(ValueError):
        preprocess.Preprocessor.for_knn().fit(table)


def test_constant_column_does_not_divide_by_zero():
    table = toy_table([1, 1, 1, 1], extra_numeric=np.array([[1.0, 5.0, 3.0]] * 4))
    out = preprocess.Preprocessor.for_knn().fit_transform(table)
    assert np.isfinite(out).all()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="data/networkTraffic.csv not present")
def test_loader_reproduces_documented_facts():
    table = data.load_table()
    assert table.n == 257673
    assert table.numeric.shape[1] == 39 and "id" not in table.numeric_names
    assert list(data.class_counts(table.y)) == [93000, 13987, 2329, 16353, 44525, 2677, 24246, 174, 1511, 58871]
    assert "?" not in table.vocab[table.nominal_names.index("service")]
    assert "none" in table.vocab[table.nominal_names.index("service")]
    assert len(table.vocab[table.nominal_names.index("proto")]) == 133
    sub = table.take(np.arange(10))
    assert sub.n == 10 and sub.nominal_values("proto").shape == (10,)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="data/networkTraffic.csv not present")
def test_full_pipelines_produce_finite_matrices():
    table = data.load_table().take(np.arange(0, 257673, 20))
    for pre in (preprocess.Preprocessor.for_knn(), preprocess.Preprocessor.for_tree()):
        out = pre.fit_transform(table)
        assert np.isfinite(out).all() and out.shape[1] == pre.n_features_
        assert out.shape[1] > 60
    assert preprocess.Preprocessor.for_knn().fit(table).n_numeric_ == 36
