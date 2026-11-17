"""export gifti read back by nibabel (the golden check of the exporter). SYNTHETIC files only.

    NXR_EXPORT_GOLDEN_DIR=<scratch> ctest -R export_gifti      # test_export_gifti keeps its files there
    NXR_EXPORT_GOLDEN_DIR=<scratch> uv run --project $NXR python -m pytest -q io/test/parity/test_export_gifti.py

The expected values are the synthetic package's own (see io/test/test_export_gifti.cpp): two tetrahedra, a map, a series of 40 frames at 2 Hz from 1 s, a labelling.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
D = os.environ.get("NXR_EXPORT_GOLDEN_DIR")
pytestmark = pytest.mark.skipif(not D, reason="set NXR_EXPORT_GOLDEN_DIR (test_export_gifti keeps its files there)")
G = Path(D) if D else Path(".")


def meta(obj):
    return dict(obj.meta)


def test_map_hemispheres():
    expect = np.array([0.1 * (v + 1) + 1e-9 for v in range(8)], dtype=np.float32)
    for hemi, rows, structure in (("L", slice(0, 4), "CortexLeft"), ("R", slice(4, 8), "CortexRight")):
        img = nib.load(G / f"map_hemi-{hemi}.shape.gii")
        assert len(img.darrays) == 1
        np.testing.assert_array_equal(img.darrays[0].data, expect[rows])
        assert meta(img)["AnatomicalStructurePrimary"] == structure
        assert meta(img)["nxr-neuro:licence"] == "CC-BY-SA-4.0"
        assert meta(img.darrays[0])["nxr-neuro:unit"] == "V"


def test_surface():
    img = nib.load(G / "map_hemi-R.surf.gii")
    xyz, tri = img.agg_data("pointset"), img.agg_data("triangle")
    base = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=np.float32)
    np.testing.assert_allclose(xyz, base + np.array([0.1, 0, 0], dtype=np.float32), atol=1e-7)
    assert tri.shape == (4, 3) and tri.min() >= 0 and tri.max() < 4


def test_series():
    img = nib.load(G / "series_all.func.gii")
    assert len(img.darrays) == 40
    data = np.stack([d.data for d in img.darrays])            # frames x vertices
    want = np.array([[v + 100.0 * t for v in range(8)] for t in range(40)], dtype=np.float32)
    np.testing.assert_array_equal(data, want)
    assert abs(float(meta(img.darrays[3])["nxr-neuro:time"]) - (1.0 + 0.5 * 3)) < 1e-9
    right = nib.load(G / "series_hemi-R.func.gii")
    assert len(right.darrays) == 3
    np.testing.assert_array_equal(np.stack([d.data for d in right.darrays]), want[1:4, 4:8])


def test_labels():
    img = nib.load(G / "atlas.label.gii")
    np.testing.assert_array_equal(img.darrays[0].data, np.array([0, 1, 1, 1, 2, 2, 2, 0], dtype=np.int32))
    table = {l.label: (l.key, l.red, l.blue) for l in img.labeltable.labels}
    assert table["alpha"][0] == 1 and abs(table["alpha"][1] - 1.0) < 1e-6 and abs(table["alpha"][2] - 128 / 255) < 1e-6
