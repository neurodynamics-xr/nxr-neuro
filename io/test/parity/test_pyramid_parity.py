"""The measurement pyramid, read back by nxr: what `nxr-neuro pyramid build` stores is what nxr's own atlas build stores (names, keys, layout), so nxr's readers find it.

Needs the nxr-neuro build (pipelines on) and nxr's environment:

    NXR=<path to nxr> NXR_NEURO_TEST_PYRAMID=<build>/pipelines/test_pyramid_pipeline \\
      uv run --project $NXR python -m pytest -q --basetemp=<a scratch folder> test/parity/test_pyramid_parity.py

The C++ pipeline test makes a SYNTHETIC dataset (never a real dataset) and keeps it (NXR_TEST_KEEP); nxr's Python then reads it through its own rows and arrays:
(a) nxr's `atlas_view` discovers the frame tiling (kind frames), the source estimate (a Field with no bytes, `source estimate`, by the kernel, from the recording) and the joint
    partition (spatiotemporal, tree, level, n_frames, n_tiles, 6 bands, 18 rows);
(b) the rows are nxr's: `params_json.atlas` {tree, level, finest, gauge, areas, node} on the leaf partition, `params_json.joint` {space, time, bits, tiles, frames, layout [band, tile, frame]},
    measures named as nxr's ATLAS_MEASURES (power_sum, envelope_sum, envelope_max), within = the `atlas bands` spans, code = the band's index, the node named as nxr's `atlas_node`;
(c) nxr reads the arrays [band, leaf, frame] through the rows, rolls them up with its own `rollup` (space by reshape, time on the tower: sums add), and the per-frame `count` is a
    count on the frame tiling.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

NXR = os.environ.get("NXR")
BIN = os.environ.get("NXR_NEURO_TEST_PYRAMID")
pytestmark = pytest.mark.skipif(not NXR or not BIN, reason="set NXR and NXR_NEURO_TEST_PYRAMID (the nxr-neuro build's test_pyramid_pipeline)")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    keep = tmp_path_factory.mktemp("pyramid") / "ds"
    p = subprocess.run([BIN], env={**os.environ, "NXR_TEST_KEEP": str(keep)}, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    from nxr.io.writer import open_writer
    ds = open_writer(keep / "demo", lock=False)
    yield ds


def test_atlas_view_finds_it(dataset):
    rows = dataset.db.all("SELECT * FROM atlas_view WHERE field = 'rest × fixed'")
    kinds = {r["kind"]: r for r in rows}
    assert set(kinds) == {"frames", "spatiotemporal"}
    st = kinds["spatiotemporal"]
    assert st["field_kind"] == "source estimate" and st["tree"] == "subject" and st["n_bands"] == 6 and st["n_rows"] == 18
    assert st["measures"] == "envelope_max,envelope_sum,power_sum" and st["n_frames"] == 122 and st["n_tiles"] == 2 ** st["level"] * 2
    assert st["recording_id"] and st["kernel_id"]
    assert kinds["frames"]["measures"] == "count"
    est = dataset.db.one("SELECT * FROM field WHERE id = ?", st["field_id"])
    assert est["path"] is None and est["by_operator_id"] == st["kernel_id"] and est["derived_from_id"] == st["recording_id"]


def test_rows_are_nxrs(dataset):
    import json
    from nxr.io.layout import atlas_node
    from nxr.pipelines.atlas.atlas_store import ATLAS_MEASURES
    sub = dataset.db.one("SELECT * FROM subject WHERE name = 'S1'")
    joint = dataset.db.one("SELECT * FROM selection WHERE subject_id = ? AND name LIKE 'rest × fixed %'", sub["id"])
    p = json.loads(joint["params_json"])
    assert p["joint"]["layout"] == ["band", "tile", "frame"] and p["joint"]["bits"] >= 7 and p["atlas"]["tree"] == "subject"
    space = dataset.db.one("SELECT * FROM selection WHERE id = ?", p["joint"]["space"])
    sp = json.loads(space["params_json"])["atlas"]
    assert {"tree", "level", "finest", "gauge", "areas", "node"} <= set(sp) and sp["node"].endswith(f"tiles_L{sp['level']}_atlas")
    bands = dataset.db.one("SELECT * FROM selection WHERE subject_id = ? AND name = 'atlas bands' AND type = 'spans'", sub["id"])
    ms = dataset.db.all("SELECT * FROM selection_measurement WHERE selection_id = ?", joint["id"])
    assert {m["measure"] for m in ms} == {ATLAS_MEASURES["power"], ATLAS_MEASURES["env"], ATLAS_MEASURES["envmax"]}
    assert {m["within_selection_id"] for m in ms} == {bands["id"]} and sorted({m["within_code"] for m in ms}) == list(range(6))
    prod = dataset.db.one("SELECT manifold_id FROM selection WHERE id = ?", joint["id"])["manifold_id"]
    assert p["atlas"]["node"] == f"{atlas_node(dataset.db, prod, 'rest__fixed')}/subject"


def test_nxr_reads_and_rolls_up(dataset):
    import json
    from nxr.measure import rollup
    from nxr.pipelines.atlas.nxr_store import SubjectStore
    sub = dataset.db.one("SELECT * FROM subject WHERE name = 'S1'")
    store = SubjectStore(dataset, sub)
    joint = dataset.db.one("SELECT * FROM selection WHERE subject_id = ? AND name LIKE 'rest_b × fixed %'", sub["id"])
    p = json.loads(joint["params_json"])
    depth = json.loads(dataset.db.one("SELECT params_json FROM selection WHERE id = ?", p["joint"]["space"])["params_json"])["atlas"]["level"]
    power = store.array(f"{p['atlas']['node']}/power")
    assert power.shape == (6, 2 ** depth * 2, p["joint"]["frames"]) and np.all(np.isfinite(power)) and (power >= 0).all()
    whole = rollup.space(power, depth, -1, axis=1)               # the whole cortex: a sum over leaves
    assert np.allclose(whole[:, 0, :], power.sum(axis=1)) if whole.ndim == 3 else np.allclose(whole, power.sum(axis=1))
    frames = dataset.db.one("SELECT * FROM selection WHERE id = ?", p["joint"]["time"])
    cnt = dataset.db.one("SELECT array_path FROM selection_measurement WHERE selection_id = ? AND measure = 'count'", frames["id"])
    samples = store.array(cnt["array_path"])
    assert samples.shape == (p["joint"]["frames"],) and samples.sum() == 10000 - 125     # every chunk, the 0.5 s bad span carries no count
    t = rollup.time(power, 2, "sum", 2, json.loads(frames["params_json"])["atlas"]["frame_code0"])
    assert np.isclose(t.sum(), power.sum())                        # time roll-up on the tower conserves a sum


def test_node_key_is_nxrs_method_stamp(dataset):
    """a Brainstorm-named kernel (`<method>_KERNEL_<stamp>`) gives nxr's key `<recording>__<method>_<stamp>` (build.py), and its arrays are float64 / int64 as nxr's"""
    import json
    from nxr.io.layout import atlas_node
    sub = dataset.db.one("SELECT * FROM subject WHERE name = 'S1'")
    est = dataset.db.one("SELECT * FROM field WHERE subject_id = ? AND kind = 'source estimate' AND name LIKE '% × MN_MEG_KERNEL_261005_2149'", sub["id"])
    pj = json.loads(est["producer_json"])
    assert (pj["method"], pj["stamp"]) == ("MN_MEG", "261005_2149") and est["data_type"] == "float32"
    joint = dataset.db.one("SELECT * FROM selection WHERE subject_id = ? AND of_field_id = ? AND json_extract(params_json, '$.joint') IS NOT NULL", sub["id"], est["id"])
    p = json.loads(joint["params_json"])
    assert p["atlas"]["node"] == f"{atlas_node(dataset.db, joint['manifold_id'], 'rest__MN_MEG_261005_2149')}/subject"
    from nxr.pipelines.atlas.nxr_store import SubjectStore
    store = SubjectStore(dataset, sub)
    assert store.array(f"{p['atlas']['node']}/power").dtype == np.float64


def test_free_orientation_tensor_is_nxrs(dataset):
    """a free kernel gives nxr's tensor: arrays `tensor/{t11..total}` [band, leaf, frame], measures as ATLAS_MEASURES, a vector3 estimate; nxr reads and rolls them up"""
    import json
    from nxr.measure import rollup
    from nxr.pipelines.atlas.atlas_store import ATLAS_MEASURES
    from nxr.pipelines.atlas.nxr_store import SubjectStore
    from nxr.measure.vectors import TENSOR_KEYS
    sub = dataset.db.one("SELECT * FROM subject WHERE name = 'S1'")
    est = dataset.db.one("SELECT * FROM field WHERE subject_id = ? AND kind = 'source estimate' AND name = 'rest × free'", sub["id"])
    assert est["value_type"] == "vector3" and est["path"] is None and est["data_type"] == "float32"
    assert json.loads(est["producer_json"])["orientation"] == "free"
    joint = dataset.db.one("SELECT * FROM selection WHERE subject_id = ? AND of_field_id = ? AND json_extract(params_json, '$.joint') IS NOT NULL", sub["id"], est["id"])
    p = json.loads(joint["params_json"])
    ms = dataset.db.all("SELECT * FROM selection_measurement WHERE selection_id = ?", joint["id"])
    assert {m["measure"] for m in ms} == {ATLAS_MEASURES[f"tensor/{k}"] for k in TENSOR_KEYS} and len(ms) == 60
    assert {m["array_path"] for m in ms} == {f"{p['atlas']['node']}/tensor/{k}" for k in TENSOR_KEYS}
    store = SubjectStore(dataset, sub)
    depth = json.loads(dataset.db.one("SELECT params_json FROM selection WHERE id = ?", p["joint"]["space"])["params_json"])["atlas"]["level"]
    arrs = {k: store.array(f"{p['atlas']['node']}/tensor/{k}") for k in TENSOR_KEYS}
    assert all(a.dtype == np.float64 and a.shape == (6, 2 ** depth * 2, p["joint"]["frames"]) for a in arrs.values())
    # a Hermitian positive-semidefinite tensor: the trace is the total where every vertex has a frame, and the total never falls below it
    tr = arrs["t11"] + arrs["t22"] + arrs["tnn"]
    assert (arrs["total"] >= tr - 1e-12).all() and (arrs["t11"] >= 0).all()
    assert np.allclose(rollup.space(arrs["total"], depth, -1, axis=1).sum(), arrs["total"].sum())
