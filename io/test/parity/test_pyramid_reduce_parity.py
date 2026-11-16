"""The GROUP REDUCE (step 6): `nxr-neuro pyramid reduce` against nxr's own `reduce_dataset` on the SAME inputs.

Needs the nxr-neuro build (pipelines on) and nxr's environment:

    NXR=<path to nxr> NXR_NEURO_TEST_REDUCE=<build>/pipelines/test_pyramid_reduce \\
      uv run --project $NXR python -m pytest -q --basetemp=<a scratch folder> test/parity/test_pyramid_reduce_parity.py

The C++ test makes a SYNTHETIC dataset (never a real dataset): a default subject (a two-hemisphere brain, its registered sphere, the group tree at depth 3) and three subjects whose surfaces and spheres differ,
each measured on the group tree (`pyramid build --tree group`, a constrained and a free kernel, two recordings for one subject, two kernels of one method for another) with a scalar map summarised on the
group partition. It keeps the dataset BEFORE the reduce (`pre`) and AFTER nxr-neuro's reduce (`post`). nxr's reduce runs on a copy of `pre`; the two default subjects are compared:
(a) the group fields: names, kind (`group map` / `source estimate`), pathless, function sum, unit, data_type, producer_json (the keys, subjects, kernels), contributions (member, source field);
(b) the rows on the default subject's partition: measure, array path, unit, within code, the group field they are of;
(c) every array: the same paths, the same shapes, the same values (relative 1e-12, NaN where NaN; maxima -inf where -inf);
(d) the partition's `atlas` block (tree group, level, node, members).
nxr's `reduce` sums every complete subject: the copy marks the two subjects the C++ test names for refusals (S4: no group tree, S5: other bands) not complete, as the caller's cohort leaves them out.
nxr's readers go through the rows (nxr PR 84: the definitions on a `set` or a `ladder` Selection, the codes at `array_path`), so nothing of nxr is patched here.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import writer_scenario as ws  # noqa: E402

NXR = os.environ.get("NXR")
BIN = os.environ.get("NXR_NEURO_TEST_REDUCE")
pytestmark = pytest.mark.skipif(not NXR or not BIN, reason="set NXR and NXR_NEURO_TEST_REDUCE (the nxr-neuro build's test_pyramid_reduce)")


@pytest.fixture(scope="module", params=[pytest.param(51, id="f51"), pytest.param(52, id="f52")])
def store_format(request):
    """the store format the C++ test makes its dataset at (and nxr's reduce then works on a copy of it)"""
    return request.param


@pytest.fixture(scope="module")
def made(tmp_path_factory, store_format):
    keep = tmp_path_factory.mktemp("reduce")
    p = subprocess.run([BIN], env={**os.environ, "NXR_TEST_KEEP": str(keep), "NXR_TEST_STORE_FORMAT": str(store_format)}, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    theirs = keep / "nxr"
    shutil.copytree(keep / "pre", theirs)
    con = sqlite3.connect(theirs / "demo" / "dataset.sqlite")
    con.execute("UPDATE subject SET status = 'pending' WHERE name IN ('S4', 'S5')")
    con.commit(); con.close()
    from nxr.pipelines.atlas import reduce as R
    r = R.reduce_dataset(theirs / "demo", sources=(), kernels="latest", log=lambda *a: None)
    assert r["members"] == 3 and r["skipped"] == []
    from nxr.io.writer import open_writer
    return open_writer(keep / "post" / "demo", lock=False), open_writer(theirs / "demo", lock=False), keep


def _default(ds):
    return ds.db.one("SELECT * FROM subject WHERE name = '@default_subject'")


def _group_fields(ds):
    d = _default(ds)
    rows = ds.db.all("SELECT * FROM field WHERE subject_id = ? AND path IS NULL AND json_extract(producer_json, '$.atlas') IN ('spatial', 'time') ORDER BY name", d["id"])
    return {r["name"]: r for r in rows}


def test_group_fields_are_nxrs(made):
    ours, theirs, _ = made
    a, b = _group_fields(ours), _group_fields(theirs)
    assert set(a) == set(b) == {"group synthetic_map", "group rest__MN_MEG__constrained", "group rest__WMNE_MEG__free"}
    for name in a:
        x, y = a[name], b[name]
        for col in ("kind", "function", "unit", "data_type", "value_type", "path", "description"):
            assert x[col] == y[col], (name, col, x[col], y[col])
        assert json.loads(x["producer_json"]) == json.loads(y["producer_json"]), name
        cx = sorted((c["subject_id"], c["source_field_id"], c["weight"]) for c in ours.db.all("SELECT * FROM field_contribution WHERE field_id = ?", x["id"]))
        cy = sorted((c["subject_id"], c["source_field_id"], c["weight"]) for c in theirs.db.all("SELECT * FROM field_contribution WHERE field_id = ?", y["id"]))
        assert cx == cy and len(cx) >= 1, name


def _rows(ds):
    d = _default(ds)
    part = ds.db.one("SELECT * FROM selection WHERE subject_id = ? AND name = 'cortex_pial tiles L3'", d["id"])
    rows = ds.db.all("SELECT m.measure, m.array_path, m.unit, m.within_code, m.code, f.name AS field, (m.within_selection_id IS NOT NULL) AS banded FROM selection_measurement m JOIN field f ON f.id = m.of_field_id "
                     "WHERE m.selection_id = ? AND f.name LIKE 'group %'", part["id"])
    return part, sorted(tuple(r.values()) if hasattr(r, "values") else tuple(r) for r in rows)


def test_rows_are_nxrs(made):
    ours, theirs, _ = made
    pa, ra = _rows(ours)
    pb, rb = _rows(theirs)
    assert ra == rb and len(ra) == 6 + 3 * 6 + 10 * 6
    a, b = json.loads(pa["params_json"])["atlas"], json.loads(pb["params_json"])["atlas"]
    for k in ("tree", "level", "node", "members"):
        assert a[k] == b[k], k
    assert a["tree"] == "group" and a["level"] == 3 and a["members"] == ["S1", "S2", "S3"]
    # the bands of the time rows are the default subject's `atlas bands`
    for ds in (ours, theirs):
        d = _default(ds)
        bands = ds.db.one("SELECT * FROM selection WHERE subject_id = ? AND name = 'atlas bands' AND type = 'spans'", d["id"])
        ids = {r["within_selection_id"] for r in ds.db.all("SELECT DISTINCT within_selection_id FROM selection_measurement WHERE selection_id = ? AND within_selection_id IS NOT NULL", pa["id"] if ds is ours else pb["id"])}
        assert ids == {bands["id"]}


def _arrays(ds, root: Path):
    d = _default(ds)
    store = root / "demo" / f"{d['name']}.nxr.zarr"
    part = ds.db.one("SELECT * FROM selection WHERE subject_id = ? AND name = 'cortex_pial tiles L3'", d["id"])
    node = json.loads(part["params_json"])["atlas"]["node"]
    from nxr.io import small_files
    out = {}
    for sub in ("maps", "time"):
        base = f"{node}/{sub}"
        for rel, doc in sorted(ws.array_docs(store).items()):            # every node of the subject, rows and files alike
            if not rel.startswith(base + "/") or doc.get("node_type") != "array":
                continue
            out[rel[len(node) + 1:]] = np.asarray(small_files.open_array(store / rel, mode="r")[...])
    return out


def test_arrays_are_nxrs(made):
    ours, theirs, keep = made
    a = _arrays(ours, keep / "post")
    b = _arrays(theirs, keep / "nxr")
    assert set(a) == set(b) and len(a) > 20, sorted(set(a) ^ set(b))
    worst = 0.0
    for k in a:
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        assert x.shape == y.shape, k
        assert np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(np.isinf(x), np.isinf(y)), k
        f = np.isfinite(x)
        if f.any():
            scale = np.abs(y[f]).max() + 1e-300
            worst = max(worst, float(np.abs(x[f] - y[f]).max() / scale))
    assert worst < 1e-12, worst
    print(f"{len(a)} arrays, worst relative difference {worst:.2e}")
