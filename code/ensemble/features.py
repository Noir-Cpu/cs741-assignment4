"""Feature matrices for tree ensembles.

Two encodings of the nominal features proto, service and state are compared:

* "onehot": the tree preset of common.preprocess (raw numeric columns plus one
  indicator column per category, 79 columns on the development set).
* "ordinal": the same raw numeric columns plus one integer column per nominal
  feature (42 columns). The integer is the frequency rank of the category in
  the training rows (0 for the most frequent), after the same grouping of rare
  protocols as in the one-hot encoding. A category unseen in the training rows
  receives the rank one above the largest rank.

Both encodings are fitted on the training rows only. Numeric columns stay
untransformed: a split depends only on the order of values, so a monotone
transformation cannot change a tree.
"""
from __future__ import annotations

import numpy as np

from common import preprocess
from common.data import Table

ENCODINGS = ("onehot", "ordinal")


class ForestFeatures:
    """Fit on a training table, transform any table to a float64 matrix."""

    def __init__(self, encoding: str = "onehot"):
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        self.encoding = encoding

    def fit(self, table: Table) -> "ForestFeatures":
        self.pre_ = preprocess.Preprocessor.for_tree().fit(table)
        if self.encoding == "ordinal":
            self.rank_ = []
            for j, lut in enumerate(self.pre_.lut_):
                mapped = lut[table.nominal[:, j]]
                counts = np.bincount(mapped[mapped >= 0], minlength=len(self.pre_.categories_[j]))
                order = np.argsort(-counts, kind="stable")
                rank = np.empty(order.size, dtype=np.int64)
                rank[order] = np.arange(order.size)
                self.rank_.append(rank)
            names = list(self.pre_.numeric_names_) + list(table.nominal_names)
            self.feature_names_ = names
        else:
            self.feature_names_ = list(self.pre_.feature_names_)
        self.n_features_ = len(self.feature_names_)
        return self

    def transform(self, table: Table) -> np.ndarray:
        if self.encoding == "onehot":
            return self.pre_.transform(table)
        n_num = self.pre_.n_numeric_
        out = np.empty((table.n, self.n_features_), dtype=np.float64)
        out[:, :n_num] = table.numeric[:, self.pre_.numeric_index_]
        for j, lut in enumerate(self.pre_.lut_):
            mapped = lut[table.nominal[:, j]]
            code = np.full(table.n, self.rank_[j].size, dtype=np.float64)
            hit = mapped >= 0
            code[hit] = self.rank_[j][mapped[hit]]
            out[:, n_num + j] = code
        return out

    def fit_transform(self, table: Table) -> np.ndarray:
        return self.fit(table).transform(table)
