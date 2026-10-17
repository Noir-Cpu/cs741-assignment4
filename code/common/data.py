"""Loading of the UNSW-NB15 network traffic data set (networkTraffic.csv).

The raw file has 257,673 rows: an identifier, 42 descriptive features (39
numeric and 3 nominal) and the target attack_cat with classes 0 to 9. Missing
values are encoded as "?" and occur only in the nominal feature service. A
missing service means that the flow carries no application-layer service, so
the symbol is loaded as the valid category "none" (Assignment 2 decision).

The loader returns a Table: numpy arrays only, row-indexable with take().
Nominal features are stored as integer codes into a fixed vocabulary, so
preprocessing statistics (category frequencies) are always computed from
training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "networkTraffic.csv"
RESULTS_COMMON = REPO_ROOT / "results" / "common"

TARGET = "attack_cat"
ID_COLUMN = "id"
NOMINAL = ("proto", "service", "state")
MISSING_TOKEN = "?"
MISSING_CATEGORY = "none"

CLASS_NAMES = ("Normal", "Reconnaissance", "Backdoor", "DoS", "Exploits",
               "Analysis", "Fuzzers", "Worms", "Shellcode", "Generic")
N_CLASSES = len(CLASS_NAMES)


@dataclass
class Table:
    """Numeric block, nominal block (integer codes) and integer target.

    numeric:      float64 array (n, p) in the column order of numeric_names
    nominal:      int32 array (n, q); column j holds codes into vocab[j]
    vocab:        tuple of q string arrays; vocab[j][code] is the category name
    y:            int64 array (n,) with classes 0 to 9
    ids:          int64 array (n,) with the original id column
    """
    numeric: np.ndarray
    numeric_names: tuple
    nominal: np.ndarray
    nominal_names: tuple
    vocab: tuple
    y: np.ndarray
    ids: np.ndarray

    @property
    def n(self) -> int:
        return self.y.shape[0]

    def take(self, index) -> "Table":
        """Rows selected by an index array or boolean mask (copies the data)."""
        return Table(self.numeric[index], self.numeric_names, self.nominal[index],
                     self.nominal_names, self.vocab, self.y[index], self.ids[index])

    def column(self, name: str) -> np.ndarray:
        """Numeric column by name."""
        return self.numeric[:, self.numeric_names.index(name)]

    def nominal_values(self, name: str) -> np.ndarray:
        """Nominal column by name as an array of strings."""
        j = self.nominal_names.index(name)
        return self.vocab[j][self.nominal[:, j]]


def load_table(path: str | Path = DATA_PATH) -> Table:
    """Load networkTraffic.csv into a Table.

    Raises ValueError when a numeric column or the target contains a missing
    value, because only service is expected to contain the missing symbol.
    """
    frame = pd.read_csv(path, na_values=[MISSING_TOKEN], keep_default_na=False,
                        low_memory=False)
    if frame[TARGET].isna().any():
        raise ValueError("the target contains missing values")
    frame["service"] = frame["service"].fillna(MISSING_CATEGORY)
    for name in NOMINAL:
        if frame[name].isna().any():
            raise ValueError(f"unexpected missing values in {name}")
    numeric_names = tuple(c for c in frame.columns
                          if c not in NOMINAL and c not in (TARGET, ID_COLUMN))
    numeric = frame[list(numeric_names)].to_numpy(dtype=np.float64)
    if np.isnan(numeric).any():
        raise ValueError("unexpected missing values in a numeric column")
    vocab, codes = [], []
    for name in NOMINAL:
        values, inverse = np.unique(frame[name].to_numpy(dtype=str), return_inverse=True)
        vocab.append(values)
        codes.append(inverse.astype(np.int32))
    return Table(numeric=numeric, numeric_names=numeric_names,
                 nominal=np.stack(codes, axis=1), nominal_names=NOMINAL,
                 vocab=tuple(vocab), y=frame[TARGET].to_numpy(dtype=np.int64),
                 ids=frame[ID_COLUMN].to_numpy(dtype=np.int64))


def class_counts(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Number of instances per class 0 to n_classes - 1."""
    return np.bincount(y, minlength=n_classes)
