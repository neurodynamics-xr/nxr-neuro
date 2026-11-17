"""export fif (source estimates) read back by MNE-Python (the golden check of the exporter). SYNTHETIC files only.

    NXR_EXPORT_GOLDEN_DIR=<scratch> ctest -R export_fif        # test_export_fif keeps its files there
    NXR_EXPORT_GOLDEN_DIR=<scratch> uv run --project $NXR python -m pytest -q io/test/parity/test_export_fif.py

The expected values are the synthetic package's own (io/test/test_export_fif.cpp): two tetrahedra as the hemispheres, a map, a series of 40 frames at 2 Hz from 1 s.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")
D = os.environ.get("NXR_EXPORT_GOLDEN_DIR")
pytestmark = pytest.mark.skipif(not D, reason="set NXR_EXPORT_GOLDEN_DIR (test_export_fif keeps its files there)")
G = Path(D) if D else Path(".")


def test_map():
    stc = mne.read_source_estimate(str(G / "map-lh.stc"), subject="synthetic")
    assert stc.data.shape == (8, 1)
    np.testing.assert_array_equal(stc.data[:, 0], np.array([0.5 * (v + 1) for v in range(8)], dtype=np.float32))
    assert [list(v) for v in stc.vertices] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert stc.tmin == 0.0 and stc.tstep == 1.0


def test_series_chunked():
    stc = mne.read_source_estimate(str(G / "series_chunked-lh.stc"), subject="synthetic")
    want = np.array([[v + 100.0 * t for t in range(40)] for v in range(8)], dtype=np.float32)
    np.testing.assert_array_equal(stc.data, want)
    assert abs(stc.tmin - 1.0) < 1e-6 and abs(stc.tstep - 0.5) < 1e-6
    assert stc.times[3] == pytest.approx(2.5)
