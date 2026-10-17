"""Fit/transform preprocessing, fitted on training rows only.

Reproduces the intent of the Assignment 2 pipelines, rewritten with numpy:

* The identifier is dropped by the loader.
* The missing service symbol is the valid category "none" (loader).
* Protocol values seen fewer than rare_threshold times in the training rows
  are grouped into one category "other" (proto only). Unseen protocol values
  at transform time also map to "other".
* One-hot encoding of proto, service and state with the categories seen in
  training. A category unseen in training encodes as all zeros.
* Distance-based preset (for k-nearest neighbours):
  drops sloss, dloss and ct_ftp_cmd (correlation of at least 0.995 with a
  retained column), applies log(1 + x) to 25 heavy-tailed non-negative
  features and standardises every numeric column to zero mean and unit
  variance. Means and standard deviations come from the training rows.
* Tree preset: numeric columns pass through unchanged, because splits depend
  only on the order of values, plus the same one-hot encoding.

The output is float64 so that threshold splits on large integer-valued
features (for example stcpb) stay exact. Distance-based models may cast to
float32 after transformation.
"""
from __future__ import annotations

import numpy as np

from .data import Table

REDUNDANT_COLUMNS = ("sloss", "dloss", "ct_ftp_cmd")

LOG_COLUMNS = (
    "dur", "sbytes", "dbytes", "rate", "sload", "dload", "spkts", "dpkts",
    "sinpkt", "dinpkt", "sjit", "djit", "smean", "dmean", "response_body_len",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "tcprtt", "synack", "ackdat",
)

RARE_THRESHOLD = 150
OTHER_CATEGORY = "other"


class Preprocessor:
    """Table to dense float64 matrix. Call fit on training rows, then transform.

    Parameters
    ----------
    drop_redundant:  remove the near-duplicate columns in REDUNDANT_COLUMNS
    log_transform:   apply log(1 + x) to LOG_COLUMNS (values must be >= 0)
    standardise:     centre and scale every numeric column (training mean/std)
    group_rare:      group rare categories of the columns in rare_columns
    rare_threshold:  a category with fewer training rows is rare
    rare_columns:    nominal columns with rare-category grouping
    """

    def __init__(self, drop_redundant: bool = False, log_transform: bool = False,
                 standardise: bool = False, group_rare: bool = True,
                 rare_threshold: int = RARE_THRESHOLD, rare_columns=("proto",)):
        self.drop_redundant = drop_redundant
        self.log_transform = log_transform
        self.standardise = standardise
        self.group_rare = group_rare
        self.rare_threshold = rare_threshold
        self.rare_columns = tuple(rare_columns)
        self._fitted = False

    @classmethod
    def for_knn(cls, **overrides) -> "Preprocessor":
        """Distance-based preset."""
        params = dict(drop_redundant=True, log_transform=True, standardise=True)
        params.update(overrides)
        return cls(**params)

    @classmethod
    def for_tree(cls, **overrides) -> "Preprocessor":
        """Tree preset: raw numeric columns and one-hot nominal columns."""
        return cls(**overrides)

    # ------------------------------------------------------------------
    def fit(self, table: Table) -> "Preprocessor":
        names = list(table.numeric_names)
        keep = [i for i, c in enumerate(names)
                if not (self.drop_redundant and c in REDUNDANT_COLUMNS)]
        self.numeric_index_ = np.array(keep, dtype=np.int64)
        self.numeric_names_ = tuple(names[i] for i in keep)
        self.log_mask_ = np.array([self.log_transform and c in LOG_COLUMNS
                                   for c in self.numeric_names_])
        block = table.numeric[:, self.numeric_index_].copy()
        if self.log_mask_.any():
            if (block[:, self.log_mask_] < 0).any():
                raise ValueError("log(1 + x) requires non-negative values")
            block[:, self.log_mask_] = np.log1p(block[:, self.log_mask_])
        if self.standardise:
            self.mean_ = block.mean(axis=0)
            std = block.std(axis=0)
            self.scale_ = np.where(std > 0, std, 1.0)
        else:
            self.mean_ = np.zeros(block.shape[1])
            self.scale_ = np.ones(block.shape[1])

        self.vocab_ = table.vocab
        self.nominal_names_ = table.nominal_names
        self.lut_, self.categories_ = [], []
        for j, name in enumerate(table.nominal_names):
            size = len(table.vocab[j])
            counts = np.bincount(table.nominal[:, j], minlength=size)
            seen = counts > 0
            lut = np.full(size, -1, dtype=np.int64)
            labels = []
            if self.group_rare and name in self.rare_columns:
                rare = seen & (counts < self.rare_threshold)
                kept = np.flatnonzero(seen & ~rare)
                for code in kept:
                    lut[code] = len(labels)
                    labels.append(str(table.vocab[j][code]))
                if rare.any():
                    other = len(labels)
                    labels.append(OTHER_CATEGORY)
                    lut[~(seen & ~rare)] = other   # rare and unseen codes
            else:
                for code in np.flatnonzero(seen):
                    lut[code] = len(labels)
                    labels.append(str(table.vocab[j][code]))
            self.lut_.append(lut)
            self.categories_.append(tuple(labels))
        self.feature_names_ = list(self.numeric_names_)
        self.feature_kinds_ = ["numeric"] * len(self.numeric_names_)
        for name, labels in zip(table.nominal_names, self.categories_):
            self.feature_names_ += [f"{name}={label}" for label in labels]
            self.feature_kinds_ += [f"onehot:{name}"] * len(labels)
        self.n_features_ = len(self.feature_names_)
        self._fitted = True
        return self

    def transform(self, table: Table) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor.fit must be called first")
        n = table.n
        out = np.zeros((n, self.n_features_), dtype=np.float64)
        block = table.numeric[:, self.numeric_index_].copy()
        if self.log_mask_.any():
            block[:, self.log_mask_] = np.log1p(block[:, self.log_mask_])
        out[:, :block.shape[1]] = (block - self.mean_) / self.scale_
        offset = block.shape[1]
        rows = np.arange(n)
        for j, lut in enumerate(self.lut_):
            col = lut[table.nominal[:, j]]
            hit = col >= 0
            out[rows[hit], offset + col[hit]] = 1.0
            offset += len(self.categories_[j])
        return out

    def fit_transform(self, table: Table) -> np.ndarray:
        return self.fit(table).transform(table)

    @property
    def n_numeric_(self) -> int:
        return len(self.numeric_names_)
