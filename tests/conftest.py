"""Make code/ importable from the tests and keep Numba quiet and bounded."""
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
