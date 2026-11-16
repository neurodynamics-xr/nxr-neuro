"""Parity of the SESSION import: `nxr-neuro convert convert|resample|badsegments|events` does what nxr's Python does, on SYNTHETIC protocols only.

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro uv run --project $NXR python -m pytest -q test/parity/test_session_parity.py

The same command runs through nxr's Python and through nxr-neuro (ids a counter, a fixed clock) into two copies of one starting dataset; every row of every table, every
zarr.json (as text) and every chunk and SHARD file (bytes) are equal, and so are the printed lines. The recording's TIMESERIES GRID is asserted explicitly: raw
[channels x samples] float32, inner chunk [1 x 2^15], shard [channels x 2^17], uncompressed; the envelope [channels x windows x 2], inner [1 x 2^12 x 2], shard [channels x 2^12 x 2].
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
IO = Path(os.environ["NXR_NEURO"]) if "NXR_NEURO" in os.environ else None
pytestmark = pytest.mark.skipif(NXR is None or IO is None, reason="set NXR and NXR_NEURO")

if NXR is not None:
    import convert_fixtures as cf  # noqa: E402
    import writer_scenario as ws  # noqa: E402
    from test_convert_parity import assert_same_dataset, lines, run_io, run_nxr, starting_pair  # noqa: E402

COND = "sub-S01_ses-01_task-rest_run-01_meg"


def both(tmp_path, *args, start=1000, a=None, b=None):
    a, b = (a, b) if a is not None else starting_pair(tmp_path)
    pa = run_nxr(*args, "--dataset", a, start=start)
    pb = run_io(*args, "--dataset", b, start=start)
    return a, b, pa, pb


def same(tmp_path, root, *extra, cond=COND, rows=10):
    a, b, pa, pb = both(tmp_path, "convert", "convert", root, "--subject", "S01", "--condition", cond, *extra)
    assert pa.returncode == pb.returncode, (pa.stdout[-1500:] + pa.stderr[-1500:], pb.stdout[-1500:] + pb.stderr[-1500:])
    assert lines(pa) == lines(pb)
    if pa.returncode == 0:
        assert_same_dataset(a, b, expect_rows=rows)
    return a, b, pa, pb


@pytest.fixture(autouse=True, params=[pytest.param(51, id="f51"), pytest.param(52, id="f52")])
def store_format(request, monkeypatch):
    """every test runs on a dataset of each store format: the starting dataset (nxr's create_dataset) takes it, and both tools then write into it"""
    monkeypatch.setenv("NXR_STORE_FORMAT", str(request.param))
    return request.param


def rec_docs(folder: Path, cond=COND):
    z = folder / "S01.nxr.zarr"
    return (json.loads(ws.read_key(z, f"sensor-time/{cond}/zarr.json")), json.loads(ws.read_key(z, f"sensor-time/{cond}_envelope/zarr.json")), z)


# ── the grid ─────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_t", [1000, 32768, 40000, 131072 + 5000, 2 * 131072 + 37856])
def test_the_timeseries_grid_is_nxrs(n_t, tmp_path):
    """shorter than a chunk (1000), exactly a chunk, between a chunk and a block, a block and a bit, two blocks and a partial one (not a multiple of 2^15 nor 2^17)"""
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=n_t, epoch=4096)
    a, b, pa, pb = same(tmp_path, root)
    raw, env, z = rec_docs(b)
    c = min(1 << 15, n_t)
    s = (1 << 17) if n_t >= (1 << 15) else c
    assert raw["shape"] == [3, n_t] and raw["data_type"] == "float32" and raw["chunk_grid"]["configuration"]["chunk_shape"] == [3, s]
    sh = raw["codecs"][0]
    assert sh["name"] == "sharding_indexed" and sh["configuration"]["chunk_shape"] == [1, c]
    assert [x["name"] for x in sh["configuration"]["codecs"]] == ["bytes"]            # UNCOMPRESSED: the renderer reads byte ranges
    nw = -(-n_t // 32)
    ec = min(4096, nw)
    assert env["shape"] == [3, nw, 2] and env["chunk_grid"]["configuration"]["chunk_shape"] == [3, ec, 2]
    assert env["codecs"][0]["configuration"]["chunk_shape"] == [1, ec, 2]
    n_shards = -(-n_t // s)
    assert len([k for k in ws.subject_keys(z) if k.startswith(f"sensor-time/{COND}/c.")]) == n_shards
    # the envelope is the raw's min and max per 32 samples
    from nxr.io import small_files
    x = np.asarray(small_files.open_array(z / "sensor-time" / COND, mode="r")[:])
    e = np.asarray(small_files.open_array(z / "sensor-time" / f"{COND}_envelope", mode="r")[:])
    np.testing.assert_array_equal(x, cf.session_data(3, n_t))
    pad = np.full((3, nw * 32 - n_t), np.nan, np.float32)
    w = np.concatenate([x, pad], axis=1).reshape(3, nw, 32)
    np.testing.assert_array_equal(e[..., 0], np.nanmin(w, axis=2))
    np.testing.assert_array_equal(e[..., 1], np.nanmax(w, axis=2))


def test_the_chunk_grid_and_envelope_selections_are_in_the_rows(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=70000, epoch=4096)
    _, b, _, _ = same(tmp_path, root)
    import sqlite3
    con = sqlite3.connect(str(b / "dataset.sqlite"))
    p = json.loads(con.execute("SELECT params_json FROM selection WHERE name = ?", (f"{COND}_chunks",)).fetchone()[0])
    assert p["tiling"]["count"] == 3 and p["tiling"]["width"] == pytest.approx(32768 / 100.0) and p["end"] == pytest.approx(700.0)
    m = con.execute("SELECT measure, array_path, component FROM selection_measurement ORDER BY component").fetchall()
    assert m == [("min", f"sensor-time/{COND}_envelope", 0), ("max", f"sensor-time/{COND}_envelope", 1)]


def test_a_partial_last_epoch_and_a_relinked_binary(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=5, n_t=5000, epoch=1024, relink=True)
    a, b, pa, pb = same(tmp_path, root)
    assert any(x["stage"] == "relinked" for x in lines(pb))


def test_a_moved_binary_is_found_beside_the_link(tmp_path):
    root = tmp_path / "bst"
    info = cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=2000, epoch=500)
    import shutil
    moved = tmp_path / "moved"
    shutil.move(str(root), str(moved))
    same(tmp_path, moved)


def test_a_truncated_binary_fails_alike(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=2000, epoch=500, truncate_last=100)
    a, b, pa, pb = both(tmp_path, "convert", "convert", root, "--subject", "S01", "--condition", COND)
    assert pa.returncode == pb.returncode == 1
    assert lines(pa)[-1] == lines(pb)[-1]


def test_an_original_format_link_is_refused_with_nxrs_message(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=500, original_format="CTF")
    a, b, pa, pb = both(tmp_path, "convert", "convert", root, "--subject", "S01", "--condition", COND)
    assert pa.returncode == pb.returncode == 1 and lines(pa)[-1] == lines(pb)[-1] and "BST-BIN" in lines(pb)[-1]["message"]
    assert_same_dataset(a, b, expect_rows=1)                                  # rolled back whole


# ── flags, events ────────────────────────────────────────────────────────────────────────────────────────────────

def test_channel_flags_with_bad_channels(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=4, n_t=900, flags=[1, -1, 1, -1])
    a, b, pa, pb = same(tmp_path, root)
    import sqlite3
    assert sqlite3.connect(str(b / "dataset.sqlite")).execute("SELECT count(*) FROM selection WHERE path = ?", (f"sensor/{COND}_flags",)).fetchone()[0] == 1


def test_all_good_flags_and_unknown_flags(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=600, flags=[1, 1, 1])
    same(tmp_path, root)
    root2 = tmp_path / "bst2"
    cf.build_session_protocol(root2, condition=COND, n_chan=3, n_t=600, flags=[1, 0, 1])
    (tmp_path / "two").mkdir()
    a, b, pa, pb = both(tmp_path / "two", "convert", "convert", root2, "--subject", "S01", "--condition", COND)
    assert pa.returncode == pb.returncode == 1 and lines(pa)[-1] == lines(pb)[-1]


def events():
    return cf.make_events([
        ("blink", [1, 0, 0], [[2.0, 3.5, 7.25]]),                                       # simple
        ("saccade", [0, 0, 1], [[1.0, 6.0], [1.5, 6.25]]),                              # extended
        ("bad_motion", [0.5, 0.5, 0.5], [[4.0], [5.5]]),                                # a single extended occurrence, 'bad'
        ("BAD_zero", [0.1, 0.2, 0.3], [[8.0, 9.0]]),                                    # simple 'bad': instantaneous, no bad segment
        ("scoped", [0, 1, 0], [[1.25, 2.5]]),
    ], channels={"scoped": [["EEG000", "EEG002"], "EEG001"]})


def test_events_simple_extended_bad_and_per_channel(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=4, n_t=1500, events=events())
    a, b, pa, pb = same(tmp_path, root)
    import sqlite3
    con = sqlite3.connect(str(b / "dataset.sqlite"))
    names = {r[0] for r in con.execute("SELECT name FROM selection WHERE of_field_id IS NOT NULL")}
    assert {f"{COND}_events", f"{COND}_bad_segments"} <= names
    p = con.execute("SELECT params_json FROM selection WHERE name = ?", (f"{COND}_events",)).fetchone()[0]
    assert json.loads(p)["channels"]


def test_an_event_naming_an_unknown_channel_fails_alike(tmp_path):
    root = tmp_path / "bst"
    ev = cf.make_events([("x", [1, 0, 0], [[1.0]])], channels={"x": [["NOPE"]]})
    cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=800, events=ev)
    a, b, pa, pb = both(tmp_path, "convert", "convert", root, "--subject", "S01", "--condition", COND)
    assert pa.returncode == pb.returncode == 1 and lines(pa)[-1] == lines(pb)[-1]


# ── imported excerpts ────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("compress", [True, False])
def test_imported_excerpts(compress, tmp_path):
    root = tmp_path / "bst"
    rng = np.random.default_rng(3)
    ex = [("block001_02", rng.normal(size=(4, 700)), {"Events": events()}), ("block002_02", rng.normal(size=(4, 40000)), {})]
    cf.build_session_protocol(root, condition="C", n_chan=4, raw=False, imported=ex, sfreq=100.0)
    same(tmp_path, root, cond="C")


# ── resample, bad segments, events: on a stored recording ──────────────────────────────────────────────────────────

def converted_pair(tmp_path, root, cond=COND, extra=()):
    """a dataset nxr's Python converted a session into, copied for both tools"""
    import shutil
    a, b = starting_pair(tmp_path)
    p = run_nxr("convert", "convert", root, "--subject", "S01", "--condition", cond, "--dataset", a, *extra, start=500, ok=True)
    shutil.rmtree(b.parent)
    shutil.copytree(a.parent, b.parent)
    return a, b


def after(tmp_path, a, b, *args, rows=20):
    pa = run_nxr(*args, "--dataset", a, start=5000)
    pb = run_io(*args, "--dataset", b, start=5000)
    assert pa.returncode == pb.returncode, (pa.stdout[-1500:] + pa.stderr[-1500:], pb.stdout[-1500:] + pb.stderr[-1500:])
    assert lines(pa) == lines(pb)
    assert_same_dataset(a, b, expect_rows=rows)
    return pa, pb


@pytest.mark.parametrize("to", [50, 60, 300, 100.5])
def test_resample_is_nxrs(to, tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=5, n_t=3000, epoch=700, sfreq=100.0)
    a, b = converted_pair(tmp_path, root)
    after(tmp_path, a, b, "convert", "resample", "--subject", "S01", "--recording", COND, "--to", to)


def test_resample_a_long_recording_with_a_name_and_a_refusal(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=140000, epoch=4096, sfreq=100.0)
    a, b = converted_pair(tmp_path, root)
    after(tmp_path, a, b, "convert", "resample", "--subject", "S01", "--recording", COND, "--to", 25, "--name", "quarter")
    pa, pb = after(tmp_path, a, b, "convert", "resample", "--subject", "S01", "--recording", COND, "--to", 25, "--name", "quarter")
    assert pb.returncode == 1 and "FileExistsError" in lines(pb)[-1]["message"]
    pa, pb = after(tmp_path, a, b, "convert", "resample", "--subject", "S01", "--recording", "nope", "--to", 25)
    assert pb.returncode == 1 and lines(pa) == lines(pb)
    pa, pb = after(tmp_path, a, b, "convert", "resample", "--subject", "S01", "--recording", COND, "--to", 33.3333)
    assert pb.returncode == 1


def test_convert_with_resample_is_nxrs(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=4, n_t=2500, epoch=600, sfreq=100.0, events=events())
    same(tmp_path, root, "--resample", "40")


def test_badsegments_go_onto_the_recording_and_its_resampled_copy(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=3, n_t=3000, epoch=700, sfreq=100.0)
    a, b = converted_pair(tmp_path, root)
    after(tmp_path, a, b, "convert", "resample", "--subject", "S01", "--recording", COND, "--to", 50)
    src = tmp_path / "bad.json"
    src.write_text(json.dumps({"recordings": [{"name": COND, "segments": [{"label": "bad_muscle", "start_s": [1.0, 7.5], "stop_s": [2.5, 8.0]},
                                                                         {"label": "bad_jump", "start_s": [12.0], "stop_s": [12.0]}, {"label": "ok", "start_s": [3], "stop_s": [4]}]}]}))
    pa, pb = after(tmp_path, a, b, "convert", "badsegments", "--subject", "S01", "--from", src)
    assert [r["bad_segments"] for r in lines(pb)[-1]["recordings"]] == [2, 2]
    pa, pb = after(tmp_path, a, b, "convert", "badsegments", "--subject", "S01", "--from", src)
    assert pb.returncode == 1 and "exists (--replace)" in lines(pb)[-1]["message"]
    after(tmp_path, a, b, "convert", "badsegments", "--subject", "S01", "--from", src, "--replace")
    src.write_text(json.dumps({"recordings": [{"name": COND, "segments": []}]}))
    after(tmp_path, a, b, "convert", "badsegments", "--subject", "S01", "--from", src, "--replace")
    src.write_text(json.dumps({"recordings": [{"name": "nope", "segments": []}]}))
    pa, pb = after(tmp_path, a, b, "convert", "badsegments", "--subject", "S01", "--from", src)
    assert pb.returncode == 1


def test_events_onto_a_converted_recording_and_its_copy(tmp_path):
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=4, n_t=3000, epoch=700, sfreq=100.0, events=events())
    a, b = converted_pair(tmp_path, root)
    after(tmp_path, a, b, "convert", "resample", "--subject", "S01", "--recording", COND, "--to", 50)
    data = root / "data"
    pa, pb = after(tmp_path, a, b, "convert", "events", "--subject", "S01", "--from", data)
    assert pb.returncode == 1                                                                  # present: refused
    pa, pb = after(tmp_path, a, b, "convert", "events", "--subject", "S01", "--from", data, "--replace")
    assert [(r["recording"], r["events"], r["event_types"]) for r in lines(pb)[-1]["recordings"]] == [(COND, 10, 5), (f"{COND}_50Hz", 10, 5)]
    (tmp_path / "empty" / "S01").mkdir(parents=True)
    pa, pb = after(tmp_path, a, b, "convert", "events", "--subject", "S01", "--from", tmp_path / "empty")
    assert "no raw link" in lines(pb)[-1]["recordings"][0]["skipped"]
