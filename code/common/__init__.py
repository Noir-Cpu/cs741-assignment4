"""Common library for the CS741 Assignment 4 experiments.

The modules implement the data handling, the evaluation protocol, the
performance measures, the statistical tests and the two baseline learners
(k-nearest neighbours and the classification tree) with numpy and Numba.
Scikit-learn appears only inside tests/ as a correctness oracle.

Import from a script as follows, with the repository root derived from the
script location:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # code/
    from common import protocol, data
"""
