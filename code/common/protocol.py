"""Shared evaluation protocol: master seed, derived random streams and partitions.

Every random stream is derived from the master seed and a named purpose
through numpy.random.SeedSequence, so that no two purposes share a stream and
every result is reproducible from the purpose name alone.

Protocol (identical for every model):

* Generalisation set G: stratified 20 percent hold-out, fixed once by the
  purpose "generalisation-split". Never used for tuning or model selection.
* Development set D: the remaining 80 percent.
* Tuning: stratified 5-fold cross-validation inside D, purpose "tuning-cv".
* Comparison: stratified 5-fold cross-validation inside D repeated 3 times
  (15 paired scores per model), purpose "comparison-cv", followed by a final
  fit on all of D and a single evaluation on G.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

MASTER_SEED = 1072417991
HOLDOUT_FRACTION = 0.2
N_FOLDS = 5
N_REPEATS = 3
N_CLASSES = 10

PURPOSE_SPLIT = "generalisation-split"
PURPOSE_TUNING = "tuning-cv"
PURPOSE_COMPARISON = "comparison-cv"
PURPOSE_SUBSAMPLE = "tuning-subsample"


def _purpose_key(purpose: str) -> tuple[int, ...]:
    """Map a purpose name to four 32-bit words with SHA-256 (stable across runs)."""
    digest = hashlib.sha256(purpose.encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[4 * i:4 * i + 4], "little") for i in range(4))


def seed_sequence(purpose: str) -> np.random.SeedSequence:
    """SeedSequence derived from the master seed and a purpose name."""
    return np.random.SeedSequence(entropy=MASTER_SEED, spawn_key=_purpose_key(purpose))


def rng(purpose: str) -> np.random.Generator:
    """Independent numpy Generator for a named purpose."""
    return np.random.Generator(np.random.PCG64(seed_sequence(purpose)))


def seed_int(purpose: str) -> int:
    """32-bit integer seed for a named purpose (for Numba kernels and tree seeds)."""
    return int(seed_sequence(purpose).generate_state(1, dtype=np.uint32)[0])


def run_seeds(purpose: str, n_runs: int) -> list[int]:
    """Integer seeds for n_runs independent runs of a stochastic model.

    Run i uses the purpose "<purpose>/run<i>", so adding runs never changes
    the seeds of earlier runs.
    """
    return [seed_int(f"{purpose}/run{i}") for i in range(n_runs)]


# ---------------------------------------------------------------------------
# Stratified partitions
# ---------------------------------------------------------------------------

def stratified_holdout(y: np.ndarray, fraction: float = HOLDOUT_FRACTION,
                       purpose: str = PURPOSE_SPLIT) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split into (development indices, hold-out indices), both sorted.

    Each class contributes round(fraction * count) instances to the hold-out,
    at least one instance and never the whole class.
    """
    y = np.asarray(y)
    gen = rng(purpose)
    held = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        n_hold = int(np.clip(round(fraction * members.size), 1, members.size - 1))
        held.append(gen.permutation(members)[:n_hold])
    held = np.sort(np.concatenate(held))
    mask = np.ones(y.size, dtype=bool)
    mask[held] = False
    return np.flatnonzero(mask), held


def stratified_fold_ids(y: np.ndarray, n_folds: int, gen: np.random.Generator) -> np.ndarray:
    """Fold identifier (0 to n_folds - 1) for every instance, stratified by class.

    Instances are ordered by class and, inside a class, at random. Walking the
    ordering and assigning folds cyclically with a random start per class gives
    each class an almost equal share of every fold, and fold sizes that differ
    by at most the number of classes.
    """
    y = np.asarray(y)
    fold = np.empty(y.size, dtype=np.int64)
    offset = 0
    for c in np.unique(y):
        members = gen.permutation(np.flatnonzero(y == c))
        start = int(gen.integers(0, n_folds))
        fold[members] = (np.arange(members.size) + start + offset) % n_folds
        offset += members.size
    return fold


@dataclass(frozen=True)
class Partition:
    """One training/test partition. Indices refer to the array that was split."""
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def stratified_kfold(y: np.ndarray, n_folds: int = N_FOLDS, purpose: str = PURPOSE_TUNING,
                     repeat: int = 0) -> list[Partition]:
    """One stratified k-fold partitioning (a single repeat)."""
    gen = rng(f"{purpose}/repeat{repeat}")
    fold = stratified_fold_ids(y, n_folds, gen)
    return [Partition(repeat, f, np.flatnonzero(fold != f), np.flatnonzero(fold == f))
            for f in range(n_folds)]


def repeated_stratified_kfold(y: np.ndarray, n_folds: int = N_FOLDS, n_repeats: int = N_REPEATS,
                              purpose: str = PURPOSE_COMPARISON) -> list[Partition]:
    """Repeated stratified k-fold: n_folds * n_repeats partitions, repeat-major order."""
    out: list[Partition] = []
    for r in range(n_repeats):
        out.extend(stratified_kfold(y, n_folds, purpose, repeat=r))
    return out


def tuning_partitions(y_dev: np.ndarray) -> list[Partition]:
    """Shared tuning folds: stratified 5-fold cross-validation inside D."""
    return stratified_kfold(y_dev, N_FOLDS, PURPOSE_TUNING)


def comparison_partitions(y_dev: np.ndarray) -> list[Partition]:
    """Shared comparison partitions: 5-fold cross-validation repeated 3 times inside D."""
    return repeated_stratified_kfold(y_dev, N_FOLDS, N_REPEATS, PURPOSE_COMPARISON)


def stratified_subsample(y: np.ndarray, n: int, purpose: str = PURPOSE_SUBSAMPLE,
                         min_per_class: int = 0) -> np.ndarray:
    """Sorted indices of a class-proportional subsample of about n instances.

    The quota of a class is n * (class share), rounded by the largest-remainder
    method, raised to min_per_class where the class has that many instances.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if n >= y.size:
        return np.arange(y.size)
    exact = n * counts / y.size
    quota = np.floor(exact).astype(np.int64)
    short = n - int(quota.sum())
    if short > 0:
        quota[np.argsort(-(exact - quota), kind="stable")[:short]] += 1
    quota = np.minimum(np.maximum(quota, np.minimum(min_per_class, counts)), counts)
    gen = rng(purpose)
    picked = [gen.permutation(np.flatnonzero(y == c))[:q] for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(picked))


def index_digest(index: np.ndarray) -> str:
    """Short SHA-256 digest of an index array, to verify that a split is unchanged."""
    return hashlib.sha256(np.ascontiguousarray(index, dtype=np.int64).tobytes()).hexdigest()[:16]
