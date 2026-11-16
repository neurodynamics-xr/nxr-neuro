"""Parity of the WRITER: nxr-neuro writes what nxr's Python writes — the same rows, the same zarr.json bytes, the same arrays — and the two
read, lock and recover each other's datasets.

Run with nxr's environment (it has zarr and the writer) and the built tools:

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro NXR_NEURO_SCENARIO=<build>/io/nxr-neuro-scenario \\
      uv run --project $NXR python -m pytest -q test/parity

Synthetic data only (an icosphere generated here). The scenario is `writer_scenario.py` (nxr's Python) and `io/test/scenario_driver.cpp`
(nxr-io's API), with a counter for ids and a fixed clock on both sides, so a row compares equal to its twin without any mapping.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import writer_scenario as ws  # noqa: E402

NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
IO = Path(os.environ["NXR_NEURO"]) if "NXR_NEURO" in os.environ else None
SCENARIO = Path(os.environ["NXR_NEURO_SCENARIO"]) if "NXR_NEURO_SCENARIO" in os.environ else None
pytestmark = pytest.mark.skipif(NXR is None or IO is None or SCENARIO is None, reason="set NXR, NXR_NEURO and NXR_NEURO_SCENARIO")

NXR_MAIN = "import sys; from nxr.cli.main import main; sys.exit(main(sys.argv[1:]))"


def run_io(*args, env=None, ok=None):
    p = subprocess.run([str(IO), *map(str, args)], capture_output=True, text=True, env={**os.environ, **(env or {})})
    if ok is True:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


def run_nxr(*args, env=None, ok=None):
    p = subprocess.run([sys.executable, "-c", NXR_MAIN, *map(str, args)], capture_output=True, text=True, env={**os.environ, **(env or {})})
    if ok is True:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


def lines(p) -> list[dict]:
    return [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]


UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z")


def mask(x, roots=()):
    """ids, timestamps, the producing tool and the temporary paths masked, so two runs compare"""
    s = json.dumps(x, sort_keys=True)
    for r in roots:
        s = s.replace(str(Path(r).resolve()), "<ROOT>").replace(str(r), "<ROOT>")
    s = UUID.sub("<ID>", s)
    s = STAMP.sub("<T>", s)
    s = re.sub(r"nxr-(convert|io|neuro) [0-9.]+ / (Python [0-9.]+|C\+\+17)", "<STAMP>", s)
    return json.loads(s)


# ── the scenario, through both ─────────────────────────────────────────────────────────────────────────────────
FORMATS = [pytest.param(51, id="f51"), pytest.param(52, id="f52")]      # the store formats the parity holds at: the folder tree and the small-file store


@pytest.fixture(params=FORMATS)
def fmt(request, monkeypatch):
    """the store format a test makes its datasets at: nxr's default (NXR_STORE_FORMAT) and the scenario driver's (NXR_SCENARIO_STORE_FORMAT) follow it"""
    monkeypatch.setenv("NXR_STORE_FORMAT", str(request.param))
    monkeypatch.setenv("NXR_SCENARIO_STORE_FORMAT", str(request.param))
    return request.param


@pytest.fixture(scope="module", params=FORMATS)
def pair(request, tmp_path_factory):
    w = tmp_path_factory.mktemp("write")
    ws.write_obj(w / "sphere.obj", 4)
    (w / "py").mkdir()
    (w / "cpp").mkdir()
    old = {k: os.environ.get(k) for k in ("NXR_STORE_FORMAT", "NXR_SCENARIO_STORE_FORMAT")}
    os.environ["NXR_STORE_FORMAT"] = os.environ["NXR_SCENARIO_STORE_FORMAT"] = str(request.param)
    try:
        ws.scenario(w / "py" / "ds", w / "sphere.obj")
        p = subprocess.run([str(SCENARIO), "scenario", str(w / "cpp" / "ds"), str(w / "sphere.obj")], capture_output=True, text=True)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    assert p.returncode == 0, p.stderr
    return w


def test_every_row_of_every_table_is_equal(pair):
    a, b = ws.dump_rows(pair / "py" / "ds" / "demo"), ws.dump_rows(pair / "cpp" / "ds" / "demo")
    assert a["_user_version"] == b["_user_version"] == 50
    for t in sorted(set(a) | set(b)):
        assert a[t] == b[t], f"table {t} differs"
    assert sum(len(v) for k, v in a.items() if isinstance(v, list)) > 50


def test_every_file_is_equal_zarr_json_text_and_chunk_bytes(pair):
    fa, fb = ws.dump_files(pair / "py" / "ds"), ws.dump_files(pair / "cpp" / "ds")
    assert sorted(fa) == sorted(fb)
    text_differs = [k for k in fa if fa[k] != fb[k] and fa[k][0] == "text"]
    assert not text_differs, f"zarr.json / catalog.json differ: {text_differs}"
    chunk_differs = [k for k in fa if fa[k] != fb[k] and fa[k][0] == "bytes"]
    assert not chunk_differs, f"compressed chunk bytes differ (the c-blosc builds differ?): {chunk_differs[:5]}"
    kinds = {Path(k).parent.name for k in fa if fa[k][0] == "bytes"}
    assert len(fa) > 30 and any("big" in k for k in fa)


def test_decoded_arrays_are_equal(pair):
    from nxr.io import small_files
    n = 0
    mine, theirs = pair / "py" / "ds" / "demo" / "S1.nxr.zarr", pair / "cpp" / "ds" / "demo" / "S1.nxr.zarr"
    for rel, doc in sorted(ws.array_docs(mine).items()):
        if doc["node_type"] != "array":
            continue
        x, y = small_files.open_array(mine / rel, mode="r")[...], small_files.open_array(theirs / rel, mode="r")[...]
        assert x.dtype == y.dtype and np.array_equal(x, y), rel
        n += 1
    assert n == 13


def test_the_scenario_exercises_blosc_shards_and_empty_chunks(pair):
    store = pair / "cpp" / "ds" / "demo" / "S1.nxr.zarr"
    docs = {k.split("/")[-1]: d for k, d in ws.array_docs(store).items()}
    keys = ws.subject_keys(store)
    names = lambda d: [c["name"] for c in d["codecs"]]
    inner = lambda d: [c["name"] for c in d["codecs"][0]["configuration"]["codecs"]] if names(d) == ["sharding_indexed"] else names(d)
    # "big" is over 4 MiB with no layout given: sharded by the default policy (format 51), its inner chunks blosc-compressed
    assert "blosc" in inner(docs["big"]) and "sharding_indexed" in names(docs["big_sharded"]) and "sharding_indexed" in names(docs["small_sharded"])
    assert sorted(k.split("/")[-1] for k in keys if k.startswith("source/big/c.")) == ["c.0.0"]       # five 513-row inner chunks in one shard
    assert "source/dense_zeros/c.0" in keys                      # dense: the chunk of zeros is written
    assert len(ws.read_key(store, "source/small_sharded/c.0.0")) > 0
    # shard 1 of big_sharded (rows 1024-2047): its inner chunks of rows 1024-1279 hold only zeros, so they are not stored — the index says so
    raw = ws.read_key(store, "source/big_sharded/c.1.0")
    index = np.frombuffer(raw[-(8 * 16 + 4):-4], "<u8").reshape(8, 2)
    empty = (index == np.uint64(2 ** 64 - 1)).all(axis=1)
    assert empty.tolist() == [True, True, False, False, False, False, False, False]
    # and the inner chunks are laid out in morton (Z) order, packed one after the other
    order = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    present = sorted((int(o), int(l)) for (o, l), e in zip(index, empty) if not e)
    assert present[0][0] == 0 and all(present[k][0] + present[k][1] == present[k + 1][0] for k in range(len(present) - 1))
    assert len(order) == 8


# ── cross-reads ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_nxr_reads_and_checks_a_dataset_nxr_io_wrote(pair):
    from nxr.io.database import open_database, text_parity
    import nxr.io
    ds = pair / "cpp" / "ds" / "demo"
    db = open_database(ds / "dataset.sqlite")
    parity = text_parity(db, ds.parent)
    assert parity["mismatched"] == 0 and parity["nodes"] > 10, parity
    d = nxr.io.open(ds)
    assert d.subject("S1") is not None
    for cmd in (["dataset", "show", "--dataset", ds], ["subject", "list", "--dataset", ds], ["subject", "show", "--dataset", ds, "--subject", "S1"]):
        a, b = lines(run_nxr(*cmd, ok=True)), lines(run_io(*cmd, ok=True))
        assert a == b, (cmd, a, b)
    assert lines(run_nxr("dataset", "list", "--datastore", ds.parent, ok=True)) == lines(run_io("dataset", "list", "--datastore", ds.parent, ok=True))


def test_nxr_io_reads_a_dataset_nxr_wrote(pair):
    import test_parity
    ds = pair / "py" / "ds" / "demo"
    seen = test_parity.check_dataset(ds)
    assert seen["arrays"] >= 13 and seen["sharded"] >= 2
    for cmd in (["dataset", "show", "--dataset", ds], ["subject", "show", "--dataset", ds, "--subject", "S1"]):
        assert lines(run_nxr(*cmd, ok=True)) == lines(run_io(*cmd, ok=True))


def test_a_later_write_by_each_opens_for_the_other(pair, tmp_path):
    """nxr writes, nxr-io adds a subject and a surface; then nxr adds another: one dataset, written by both"""
    shutil.copytree(pair / "py" / "ds", tmp_path / "ds")
    d = tmp_path / "ds" / "demo"
    run_io("subject", "create", "--dataset", d, "--name", "S2", ok=True)
    run_io("surface", "add", "--dataset", d, "--subject", "S2", pair / "sphere.obj", "--name", "sph", ok=True)
    run_nxr("subject", "create", "--dataset", d, "--name", "S3", ok=True)
    run_nxr("surface", "add", "--dataset", d, "--subject", "S3", pair / "sphere.obj", "--name", "sph", ok=True)
    from nxr.io.database import open_database, text_parity
    db = open_database(d / "dataset.sqlite")
    assert text_parity(db, d.parent)["mismatched"] == 0
    assert [r["name"] for r in lines(run_io("subject", "list", "--dataset", d))[0]["subjects"]] == ["S1", "S2", "S3"]
    assert lines(run_io("subject", "list", "--dataset", d)) == lines(run_nxr("subject", "list", "--dataset", d))
    from nxr.io import small_files
    a = small_files.open_array(d / "S2.nxr.zarr/source/sph/vertices", mode="r")[...]
    b = small_files.open_array(d / "S3.nxr.zarr/source/sph/vertices", mode="r")[...]
    assert np.array_equal(a, b)
    cat = json.loads((d.parent / "catalog.json").read_text())
    assert cat["datasets"][0]["subjects"] == 3


# ── the command line, verb for verb ─────────────────────────────────────────────────────────────────────────────
def test_the_commands_print_what_nxr_prints(tmp_path, fmt):
    ws.write_obj(tmp_path / "bunny.obj", 2)
    outs = {}
    for who, run in (("nxr", run_nxr), ("io", run_io)):
        root = tmp_path / who
        ds, d = root / "store", root / "store" / "demo"
        steps = [["dataset", "create", "--datastore", ds, "--name", "demo", "--source-tool", "t", "--store-format", fmt],
                 ["dataset", "create", "--datastore", ds, "--name", "demo", "--store-format", fmt],               # refused: already a dataset
                 ["dataset", "create", "--datastore", ds, "--name", "../x", "--store-format", fmt],               # refused: invalid name
                 ["dataset", "list", "--datastore", ds], ["dataset", "show", "--dataset", d],
                 ["subject", "create", "--dataset", d, "--name", "A"], ["subject", "create", "--dataset", d, "--name", "A"],
                 ["subject", "create", "--dataset", d, "--name", "T", "--kind", "template"],
                 ["subject", "create", "--dataset", d, "--name", "B"],
                 ["subject", "list", "--dataset", d], ["subject", "show", "--dataset", d, "--subject", "A"],
                 ["subject", "show", "--dataset", d, "--subject", "nobody"],
                 ["surface", "add", "--dataset", d, "--subject", "A", tmp_path / "bunny.obj"],
                 ["surface", "add", "--dataset", d, "--subject", "A", tmp_path / "bunny.obj"],      # refused: the node is taken
                 ["surface", "add", "--dataset", d, "--subject", "A", tmp_path / "bunny.obj", "--name", "second", "--scale", "0.001", "--no-primary"],
                 ["subject", "show", "--dataset", d, "--subject", "A"],
                 ["subject", "delete", "--dataset", d, "--subject", "A", "--node", "source/second"],
                 ["subject", "delete", "--dataset", d, "--subject", "A", "--node", "source/nothing"],
                 ["subject", "delete", "--dataset", d, "--subject", "B"],
                 ["dataset", "recover", "--dataset", d], ["dataset", "show", "--dataset", d],
                 ["dataset", "delete", "--datastore", ds, "--name", "demo"], ["dataset", "list", "--datastore", ds]]
        res = []
        for s in steps:
            p = run(*s)
            res.append((p.returncode, mask(lines(p), [root])))
        outs[who] = res
    for k, (a, b) in enumerate(zip(outs["nxr"], outs["io"])):
        assert a == b, f"step {k}: nxr {a} != nxr-io {b}"
    assert [r[0] for r in outs["io"]].count(1) == 6


def test_surface_add_reads_obj_ply_stl_like_nxr(tmp_path, fmt):
    """the three mesh formats (quads, negative indices, ascii and binary, both endians) give nxr's vertices, faces and rows"""
    v, f = ws.icosphere(2)
    V = np.array(v)
    F = np.array(f)
    files = {}
    # OBJ with a quad pair, texture/normal indices and negative indices
    obj = tmp_path / "quads.obj"
    with open(obj, "w") as fh:
        fh.write("# cube\n")
        cube = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        for p in cube:
            fh.write("v %r %r %r\n" % p)
        for q in [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]:
            fh.write("f " + " ".join(f"{i}/{i}/{i}" for i in q) + "\n")
    files["quads.obj"] = obj
    ply_a, ply_l, ply_b = tmp_path / "ascii.ply", tmp_path / "le.ply", tmp_path / "be.ply"
    head = lambda fmt: (f"ply\nformat {fmt} 1.0\ncomment synthetic\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
                        f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(ply_a, "w") as fh:
        fh.write(head("ascii"))
        for p in V.astype(np.float32):
            fh.write("%r %r %r\n" % tuple(float(x) for x in p))
        for t in F:
            fh.write("3 %d %d %d\n" % tuple(t))
    for path, ply_fmt, e in ((ply_l, "binary_little_endian", "<"), (ply_b, "binary_big_endian", ">")):
        with open(path, "wb") as fh:
            fh.write(head(ply_fmt).encode())
            fh.write(V.astype(e + "f4").tobytes())
            fh.write(np.concatenate([np.full((len(F), 1), 3, np.uint8).view(np.uint8), np.zeros((len(F), 0), np.uint8)], axis=1).tobytes() if False else b"".join(
                bytes([3]) + t.astype(e + "i4").tobytes() for t in F))
    files.update({"ascii.ply": ply_a, "le.ply": ply_l, "be.ply": ply_b})
    stl_b, stl_a = tmp_path / "bin.stl", tmp_path / "asc.stl"
    tri = V.astype(np.float32)[F]
    with open(stl_b, "wb") as fh:
        fh.write(b"\0" * 80 + np.uint32(len(F)).tobytes())
        for t in tri:
            fh.write(np.zeros(3, "<f4").tobytes() + t.astype("<f4").tobytes() + b"\0\0")
    with open(stl_a, "w") as fh:
        fh.write("solid s\n")
        for t in tri:
            fh.write("facet normal 0 0 0\nouter loop\n" + "".join("vertex %r %r %r\n" % tuple(float(x) for x in p) for p in t) + "endloop\nendfacet\n")
        fh.write("endsolid s\n")
    files.update({"bin.stl": stl_b, "asc.stl": stl_a})
    from nxr.io import small_files
    for name, path in files.items():
        got = {}
        for who, run in (("nxr", run_nxr), ("io", run_io)):
            root = tmp_path / ("out_" + who + "_" + name)
            d = root / "demo"
            run("dataset", "create", "--datastore", root, "--name", "demo", "--store-format", fmt, ok=True)
            run("subject", "create", "--dataset", d, "--name", "S", ok=True)
            p = run("surface", "add", "--dataset", d, "--subject", "S", path, ok=True)
            st = d / "S.nxr.zarr" / "source" / ssub(path)
            got[who] = (lines(p)[0], small_files.open_array(st / "vertices", mode="r")[...], small_files.open_array(st / "faces", mode="r")[...],
                        ws.dump_rows(d))
        a, b = got["nxr"], got["io"]
        assert mask(a[0]) == mask(b[0]), name
        assert a[1].dtype == b[1].dtype == np.float64 and np.array_equal(a[1], b[1]), name
        assert a[2].dtype == b[2].dtype == np.int32 and np.array_equal(a[2], b[2]), name
        for t in ("manifold", "topology", "topology_cell", "geometry", "space", "coordinate_system"):
            assert mask(a[3][t]) == mask(b[3][t]), (name, t)


def ssub(path: Path) -> str:
    n = path.name
    for suf in (".obj", ".ply", ".stl"):
        if n.endswith(suf):
            return n[: -len(suf)]
    return n


# ── js_number ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_js_number_is_nxrs():
    from nxr.io.database import js_number
    rng = np.random.default_rng(0)
    xs = [0.0, -0.0, 1.0, 0.1, 100.0, 1e-7, 1e-6, 1.5e-6, 1e21, 1e20, 123456789012345680000.0, 5e-324, 1.7976931348623157e308, 0.30000000000000004, -2.5, 1 / 3,
          2.0 ** 53, 4.35, 0.000123]
    xs += list(rng.normal(size=300) * 10.0 ** rng.integers(-12, 25, 300))
    p = subprocess.run([str(SCENARIO), "jsnum"], input="\n".join(repr(float(x)) for x in xs) + "\n", capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    got = p.stdout.split()
    assert got == [js_number(float(x)) for x in xs]


# ── the writer lock ─────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def fresh(tmp_path):
    d = tmp_path / "ds" / "demo"
    run_io("dataset", "create", "--datastore", tmp_path / "ds", "--name", "demo", ok=True)
    return d


def test_nxr_io_refuses_while_nxr_holds_the_lock(fresh):
    from nxr.io.writer_lock import acquire_writer_lock
    held = acquire_writer_lock(fresh)
    try:
        for cmd in (["subject", "create", "--dataset", fresh, "--name", "X"], ["dataset", "recover", "--dataset", fresh]):
            p = run_io(*cmd)
            assert p.returncode == 1
            msg = lines(p)[0]["message"]
            assert msg.startswith("WriterLockHeld: ") and "another writer holds this dataset" in msg and f"pid {os.getpid()}" in msg
            q = run_nxr(*cmd)             # and nxr, run as a process of its own, says the same thing word for word
            assert q.returncode == 1 and lines(q) == lines(p)
        assert run_io("dataset", "show", "--dataset", fresh).returncode == 0           # a reader takes no lock
        assert run_io("dataset", "delete", "--datastore", fresh.parent, "--name", "demo").returncode == 1
        assert fresh.exists()
    finally:
        held.release()
    assert run_io("subject", "create", "--dataset", fresh, "--name", "X").returncode == 0
    assert not (fresh / ".writer.lock").exists()


def test_nxr_refuses_while_nxr_io_holds_the_lock(fresh):
    from nxr.io.writer import open_writer
    from nxr.io.writer_lock import WriterLockHeld, read_writer_lock
    p = subprocess.Popen([str(SCENARIO), "hold", str(fresh)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        h = read_writer_lock(fresh)                      # nxr reads nxr-io's file
        assert h["pid"] == p.pid and h["host"] == socket.gethostname() and h["program"] == "nxr-io" and h["started_utc"].endswith("Z")
        with pytest.raises(WriterLockHeld):
            open_writer(fresh)
        q = run_nxr("subject", "create", "--dataset", fresh, "--name", "X")
        assert q.returncode == 1 and "WriterLockHeld" in lines(q)[0]["message"] and f"pid {p.pid}" in lines(q)[0]["message"]
        assert run_nxr("dataset", "show", "--dataset", fresh).returncode == 0
    finally:
        p.stdin.write("\n")
        p.stdin.flush()
        p.wait(timeout=10)
    assert not (fresh / ".writer.lock").exists()
    open_writer(fresh).close()


def test_a_dead_holders_lock_is_taken_over_and_another_hosts_is_not(fresh):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock = fresh / ".writer.lock"
    lock.write_text(json.dumps({"pid": dead.pid, "host": socket.gethostname(), "program": "nxr-convert", "started_utc": "2026-11-10T00:00:00Z"}))
    assert run_io("subject", "create", "--dataset", fresh, "--name", "X").returncode == 0           # stale: taken over
    lock.write_text(json.dumps({"pid": 1, "host": "another-host", "program": "nxr-convert", "started_utc": "2026-11-10T00:00:00Z"}))
    p = run_io("subject", "create", "--dataset", fresh, "--name", "Y")
    assert p.returncode == 1 and "another-host" in lines(p)[0]["message"]                            # never judged stale
    lock.write_text("")
    os.utime(lock, (time.time() - 60, time.time() - 60))
    assert run_io("subject", "create", "--dataset", fresh, "--name", "Y").returncode == 0           # unreadable and old: stale
    assert not lock.exists()


def test_a_borrowed_lock_is_neither_removed_nor_used_to_recover(pair, tmp_path):
    """the handshake: the holder hands its lock to the writer it runs. The writer works under it, removes nothing, and recovers nothing — a row
    pending since is the holder's live write"""
    from nxr.io.writer_lock import acquire_writer_lock
    import sqlite3
    shutil.copytree(pair / "py" / "ds", tmp_path / "ds")
    d = tmp_path / "ds" / "demo"
    held = acquire_writer_lock(d)
    try:
        c = sqlite3.connect(d / "dataset.sqlite")
        c.execute("UPDATE field SET status = 'pending' WHERE name = 'extra'")
        c.commit()
        c.close()
        env = {"NXR_WRITER_LOCK_HELD": f"{d / '.writer.lock'}:{os.getpid()}"}
        assert run_io("subject", "create", "--dataset", d, "--name", "X", env=env).returncode == 0
        assert (d / ".writer.lock").exists()                                                       # still the holder's
        assert pending(d) == 1                                                                     # not rolled back
        assert run_nxr("subject", "create", "--dataset", d, "--name", "Y", env=env).returncode == 0   # nxr does the same
        assert pending(d) == 1
    finally:
        held.release()
    assert run_io("subject", "create", "--dataset", d, "--name", "Z").returncode == 0               # the taker recovers
    assert pending(d) == 0


def test_a_read_only_database_is_refused_before_anything_is_written(fresh):
    if os.geteuid() == 0:
        pytest.skip("root can write anything")
    os.chmod(fresh / "dataset.sqlite", 0o444)
    try:
        p, q = run_io("subject", "create", "--dataset", fresh, "--name", "X"), run_nxr("subject", "create", "--dataset", fresh, "--name", "X")
        assert p.returncode == q.returncode == 1
        a, b = lines(p)[0]["message"], lines(q)[0]["message"]
        assert a.startswith("DatasetNotWritable: ") and "(EACCES)" in a and a == b
        assert not (fresh / ".writer.lock").exists()                                              # the lock was let go
    finally:
        os.chmod(fresh / "dataset.sqlite", 0o644)


# ── recovery ────────────────────────────────────────────────────────────────────────────────────────────────────
def pending(dataset: Path) -> int:
    d = ws.dump_rows(dataset)
    return sum(1 for t in ("manifold", "selection", "field", "operator", "subject") for r in d[t] if r["status"] != "complete")


@pytest.mark.parametrize("writer", ["nxr", "io"])
def test_recover_rolls_back_an_interrupted_write_as_nxr_does(pair, tmp_path, writer):
    base = tmp_path / "base"
    shutil.copytree(pair / "py" / "ds", base)
    d = base / "demo"
    if writer == "nxr":
        ws.interrupt(d, "S1")
    else:
        assert subprocess.run([str(SCENARIO), "interrupt", str(d), "S1"], capture_output=True, text=True).returncode == 0
    from nxr.io import small_files
    assert pending(d) == 1 and small_files.exists(d / "S1.nxr.zarr/source/interrupted")
    # a node left 'deleting' as well
    import sqlite3
    c = sqlite3.connect(d / "dataset.sqlite")
    c.execute("UPDATE field SET status = 'deleting' WHERE name = 'extra'")
    c.commit()
    c.close()
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(base, a)
    shutil.copytree(base, b)
    pa = run_nxr("dataset", "recover", "--dataset", a / "demo", ok=True)
    pb = run_io("dataset", "recover", "--dataset", b / "demo", ok=True)
    assert lines(pa) == lines(pb) == [{"stage": "result", "rolled_back": 1, "finished": 1}]
    assert pending(b / "demo") == 0 and not small_files.exists(b / "demo/S1.nxr.zarr/source/interrupted") and not small_files.exists(b / "demo/S1.nxr.zarr/source/extra")
    assert ws.dump_rows(a / "demo") == ws.dump_rows(b / "demo")
    assert ws.dump_files(a) == ws.dump_files(b)
    # opening for writing recovers too (the taker of the lock): a third copy, recovered by open
    c3 = tmp_path / "c"
    shutil.copytree(base, c3)
    assert run_io("subject", "create", "--dataset", c3 / "demo", "--name", "Z").returncode == 0
    assert pending(c3 / "demo") == 0


def test_a_failed_composition_rolls_back_what_it_created(pair, tmp_path):
    """a surface that cannot be added (the node is taken) leaves no pending row, no folder, no stray acquisition"""
    shutil.copytree(pair / "cpp" / "ds", tmp_path / "ds")
    d = tmp_path / "ds" / "demo"
    before = (ws.dump_rows(d), ws.dump_files(d.parent))
    assert run_io("surface", "add", "--dataset", d, "--subject", "S1", pair / "sphere.obj").returncode == 1
    after = (ws.dump_rows(d), ws.dump_files(d.parent))
    for t in before[0]:
        if t != "sync_job":
            assert before[0][t] == after[0][t], t
    assert before[1] == after[1]


try:
    from nxr.io import small_files as sf  # nxr's helper for a key of either format
except ImportError:       # run without nxr's environment: the tests are skipped
    sf = None


# ── the delete rule (D164) ──────────────────────────────────────────────────────────────────────────────────────
def build_delete_dataset(root: Path) -> Path:
    from nxr.io.writer import SubjectWriter, create_dataset
    ws.deterministic()
    ds = create_dataset(root, "demo", created_by="parity")
    sub = SubjectWriter.create(ds, "S", created_by="parity")
    sid = sub.manifold(name="m", path="source/m", type="surface", n_vertices=4, topology={"kind": "simplicial", "rank": 2, "cells": [(2, "source/m/faces")]},
                       geometry={"form": "stored", "cs_key": "scs", "positions_path": "source/m/vertices"},
                       populate=lambda at: (sub.write_array("source/m/vertices", np.zeros((4, 3))), sub.write_array("source/m/faces", np.zeros((2, 3), np.int32))))
    for n in ("a", "c", "d"):
        sub.field(name=n, path=f"source/{n}", kind="map", manifold_id=sid, data=np.arange(4, dtype=np.float32))
    # a selection that names field a's array as its weights (a row of another kind naming a path)
    sub.selection(name="w", path="source/w", type="kernel", manifold_id=sid, weight_fn="heat", params_json='{"weights":"source/a"}')
    sub.complete()
    ds.close()
    return root / "demo"


def delete_node(who, d, node):
    return (run_nxr if who == "nxr" else run_io)("subject", "delete", "--dataset", d, "--subject", "S", "--node", node, ok=True)


def test_deleting_a_field_removes_its_arrays(tmp_path, fmt):
    build_delete_dataset(tmp_path / "base")
    for who in ("nxr", "io"):
        shutil.copytree(tmp_path / "base", tmp_path / who)
        assert sf.exists(tmp_path / who / "demo/S.nxr.zarr/source/c/c.0")
        delete_node(who, tmp_path / who / "demo", "source/c")
        assert not sf.exists(tmp_path / who / "demo/S.nxr.zarr/source/c")
        assert sf.exists(tmp_path / who / "demo/S.nxr.zarr/source/d/c.0")
    assert ws.dump_rows(tmp_path / "nxr" / "demo") == ws.dump_rows(tmp_path / "io" / "demo")
    assert ws.dump_files(tmp_path / "nxr") == ws.dump_files(tmp_path / "io")


def test_an_array_another_row_names_stays(tmp_path, fmt):
    """D164/D167: a row's delete removes only the arrays no remaining row names. nxr 1.7.0 follows the same rule as nxr-io: both keep the array, and
    the two stores are equal."""
    build_delete_dataset(tmp_path / "base")
    for who in ("nxr", "io"):
        shutil.copytree(tmp_path / "base", tmp_path / who)
        delete_node(who, tmp_path / who / "demo", "source/a")
    for who in ("nxr", "io"):
        a = tmp_path / who / "demo/S.nxr.zarr/source/a"
        assert sf.exists(a / "zarr.json") and sf.exists(a / "c.0")      # the array a selection's weights name stays
        from nxr.io.database import open_database
        db = open_database(tmp_path / who / "demo" / "dataset.sqlite")
        assert db.one("SELECT 1 FROM field WHERE name = 'a'") is None and db.one("SELECT 1 FROM selection WHERE name = 'w'") is not None
    assert ws.dump_rows(tmp_path / "nxr" / "demo") == ws.dump_rows(tmp_path / "io" / "demo")
    assert ws.dump_files(tmp_path / "nxr") == ws.dump_files(tmp_path / "io")
    # once the naming row goes, nothing garbage-collects the array, and the delete of w removes w's own node (both)
    for who in ("nxr", "io"):
        delete_node(who, tmp_path / who / "demo", "source/w")
        assert not sf.exists(tmp_path / who / "demo/S.nxr.zarr/source/w")
    assert ws.dump_files(tmp_path / "nxr") == ws.dump_files(tmp_path / "io")


def test_deleting_a_manifold_and_a_subject_is_nxrs(tmp_path, fmt):
    build_delete_dataset(tmp_path / "base")
    for who in ("nxr", "io"):
        shutil.copytree(tmp_path / "base", tmp_path / who)
        # the manifold goes with its fields; w names a (now gone with its field), nothing remains to protect it
        delete_node(who, tmp_path / who / "demo", "source/m")
    ra, rb = ws.dump_rows(tmp_path / "nxr" / "demo"), ws.dump_rows(tmp_path / "io" / "demo")
    assert ra == rb
    assert not sf.exists(tmp_path / "io/demo/S.nxr.zarr/source/m")
    for who in ("nxr", "io"):
        (run_nxr if who == "nxr" else run_io)("subject", "delete", "--dataset", tmp_path / who / "demo", "--subject", "S", ok=True)
    assert ws.dump_rows(tmp_path / "nxr" / "demo") == ws.dump_rows(tmp_path / "io" / "demo")
    assert ws.dump_files(tmp_path / "nxr") == ws.dump_files(tmp_path / "io")
    assert not (tmp_path / "io/demo/S.nxr.zarr").exists()


# ── the schema ──────────────────────────────────────────────────────────────────────────────────────────────────
def _without_statistic(sql: bytes) -> bytes:
    """the schema without the statistic block (docs/specs/2026-11-13-statistics.md): proposed for the shared model, started in nxr-neuro (maintainers,
    2026-11-13); until nxr adopts it the rest stays byte for byte, and sync_job's table list is compared without 'statistic'"""
    a, b = sql.find("-- \u2500\u2500 statistic (begin".encode()), sql.find("-- \u2500\u2500 statistic (end)".encode())
    if a >= 0 and b > a:
        sql = sql[:a] + sql[sql.find(b"\n", b) + 1:].lstrip(b"\n")
    return sql.replace(b"'field', 'operator', 'statistic'))", b"'field', 'operator'))")


def test_the_embedded_schema_is_nxrs_byte_for_byte():
    ours = (Path(__file__).parents[2] / "schema" / "model.sql").read_bytes()
    theirs = (NXR / "src" / "nxr" / "io" / "model.sql").read_bytes()
    assert _without_statistic(ours) == _without_statistic(theirs)
    p = run_io("--version")
    assert "database schema 50" in p.stdout
