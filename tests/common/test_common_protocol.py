"""Tests for common.protocol: seeds, streams and stratified partitions."""
import numpy as np
import pytest

from common import protocol


def make_labels():
    gen = np.random.default_rng(protocol.seed_int("test-labels"))
    counts = [500, 300, 120, 60, 25, 7]
    return gen.permutation(np.repeat(np.arange(len(counts)), counts))


def test_master_seed_value():
    assert protocol.MASTER_SEED == 1072417991


def test_streams_are_reproducible_and_distinct():
    a = protocol.rng("alpha").random(5)
    assert np.array_equal(a, protocol.rng("alpha").random(5))
    assert not np.array_equal(a, protocol.rng("beta").random(5))
    assert protocol.seed_int("alpha") == protocol.seed_int("alpha")
    assert protocol.seed_int("alpha") != protocol.seed_int("beta")


def test_stream_depends_on_master_seed_through_seed_sequence():
    expected = np.random.SeedSequence(entropy=protocol.MASTER_SEED,
                                      spawn_key=protocol._purpose_key("alpha"))
    assert np.array_equal(expected.generate_state(4), protocol.seed_sequence("alpha").generate_state(4))


def test_run_seeds_are_stable_when_runs_are_added():
    assert protocol.run_seeds("model", 3) == protocol.run_seeds("model", 5)[:3]
    assert len(set(protocol.run_seeds("model", 10))) == 10


def test_generic_seeds_are_not_produced():
    forbidden = {0, 1, 7, 42, 123, 1234}
    assert not forbidden & set(protocol.run_seeds("model", 50))


def test_holdout_is_stratified_disjoint_and_fixed():
    y = make_labels()
    dev, gen = protocol.stratified_holdout(y)
    assert np.intersect1d(dev, gen).size == 0
    assert np.array_equal(np.sort(np.concatenate([dev, gen])), np.arange(y.size))
    for c in np.unique(y):
        n_c = (y == c).sum()
        assert abs((y[gen] == c).sum() - 0.2 * n_c) <= 1
    dev2, gen2 = protocol.stratified_holdout(y)
    assert np.array_equal(gen, gen2) and np.array_equal(dev, dev2)


def test_holdout_keeps_at_least_one_instance_per_class_on_each_side():
    y = np.array([0] * 50 + [1] * 2)
    dev, gen = protocol.stratified_holdout(y)
    assert (y[gen] == 1).sum() == 1 and (y[dev] == 1).sum() == 1


def test_kfold_partitions_cover_once_and_balance_classes():
    y = make_labels()
    parts = protocol.stratified_kfold(y, 5, "test-cv")
    assert len(parts) == 5
    all_test = np.sort(np.concatenate([p.test for p in parts]))
    assert np.array_equal(all_test, np.arange(y.size))
    for p in parts:
        assert np.intersect1d(p.train, p.test).size == 0
        assert p.train.size + p.test.size == y.size
        for c in np.unique(y):
            n_c = (y == c).sum()
            assert abs((y[p.test] == c).sum() - n_c / 5) <= 1.0


def test_repeated_kfold_gives_fifteen_distinct_partitions():
    y = make_labels()
    parts = protocol.repeated_stratified_kfold(y, 5, 3, "test-cv")
    assert len(parts) == 15
    assert [p.repeat for p in parts] == [0] * 5 + [1] * 5 + [2] * 5
    assert not np.array_equal(parts[0].test, parts[5].test)
    again = protocol.repeated_stratified_kfold(y, 5, 3, "test-cv")
    assert all(np.array_equal(a.test, b.test) for a, b in zip(parts, again))


def test_shared_partitions_use_the_documented_purposes():
    y = make_labels()
    assert len(protocol.tuning_partitions(y)) == 5
    assert len(protocol.comparison_partitions(y)) == 15
    t0 = protocol.tuning_partitions(y)[0].test
    c0 = protocol.comparison_partitions(y)[0].test
    assert not np.array_equal(t0, c0)


def test_stratified_subsample_is_proportional_and_respects_minimum():
    y = make_labels()
    sub = protocol.stratified_subsample(y, 200)
    assert abs(sub.size - 200) <= 1
    for c in np.unique(y):
        assert abs((y[sub] == c).sum() - 200 * (y == c).mean()) <= 1.01
    sub = protocol.stratified_subsample(y, 200, min_per_class=5)
    assert all((y[sub] == c).sum() >= 5 for c in np.unique(y))
    assert np.array_equal(protocol.stratified_subsample(y, 10 ** 6), np.arange(y.size))


def test_index_digest_changes_with_content():
    assert protocol.index_digest(np.arange(5)) == protocol.index_digest(np.arange(5))
    assert protocol.index_digest(np.arange(5)) != protocol.index_digest(np.arange(6))
