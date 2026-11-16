"""The conformance scenario through nxr's Python, with a deterministic clock and ids — the twin of io/test/scenario_driver.cpp.

Shared by test_write_parity.py. Importing this needs nxr (run with `uv run --project $NXR`).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import warnings
from pathlib import Path

import numpy as np

COUNTER = [0]


def det_uuid7(now_ms=None):
    COUNTER[0] += 1
    return f"00000000-0000-7000-8000-{COUNTER[0]:012x}"


def det_now(ms: bool = True):
    return "2026-01-01T00:00:00.000Z" if ms else "2026-01-01T00:00:00Z"


def deterministic(start: int = 0):
    """Make nxr's ids a counter and its clock fixed — exactly nxr::io::testing::deterministic."""
    COUNTER[0] = start
    import nxr.convert.surface as surf
    import nxr.io.database as database
    import nxr.io.writer as writer
    for mod in (database, writer):
        mod.uuid7 = det_uuid7
        mod.now_utc = det_now
    surf.now_utc = det_now


def icosphere(level: int):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [tuple(x / math.sqrt(sum(y * y for y in p)) for x in p) for p in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
         (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                p = tuple((v[a][i] + v[b][i]) / 2 for i in range(3))
                n = math.sqrt(sum(x * x for x in p))
                v.append(tuple(x / n for x in p))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return v, f


def write_obj(path: Path, level: int = 4) -> None:
    v, f = icosphere(level)
    with open(path, "w") as fh:
        for p in v:
            fh.write("v " + " ".join(repr(x) for x in p) + "\n")
        for a, b, c in f:
            fh.write(f"f {a + 1} {b + 1} {c + 1}\n")


def ramp(n: int, zero_from=-1, zero_to=-1, row_len=1) -> np.ndarray:
    i = np.arange(n, dtype=np.uint64)
    x = (i * np.uint64(1664525) + np.uint64(1013904223)) & np.uint64(0xFFFFFFFF)
    v = (x >> np.uint64(20)).astype(np.float32) / np.float32(8.0)
    if zero_from >= 0:
        row = i // np.uint64(row_len)
        v[(row >= zero_from) & (row < zero_to)] = 0
    return v


def scenario(datastore: Path, obj: Path) -> None:
    from nxr.io.mesh import read_mesh
    from nxr.io.writer import SubjectWriter, array_meta, create_dataset
    from nxr.convert.surface import add_surface
    deterministic()
    ds = create_dataset(datastore, "demo", source_tool="parity-tool", source_protocol="p", source_path="s", created_by="parity")
    sub = SubjectWriter.create(ds, "S1", created_by="parity")
    vertices, faces = read_mesh(obj)
    nv = vertices.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = add_surface(sub, name="sphere", vertices=vertices, faces=faces, source="sphere.obj", primary=True)
    sid = info["id"]
    label = "ses-01_task-rest_run-01"
    sub.acquisition(label)
    tid = sub.manifold(name="t", path="time/t", type="line", n_vertices=100, topology={"kind": "path", "rank": 1, "n_components": 1},
                       geometry={"form": "closed_form", "cs_key": "clock",
                                 "dimensions": [{"name": "time", "n_vertices": 100, "spacing": 0.001, "origin": 0.0, "unit": "s"}]}, n_edges=99)
    sub.product([sid, tid])

    small = ramp(nv)
    small_id = sub.field(name="small", path="source/small", kind="map", manifold_id=sid, data=small, acquisition=label)
    big = ramp(nv * 1000, 1000, 1500, 1000).reshape(nv, 1000)
    sub.field(name="big", path="source/big", kind="map", manifold_id=sid, data=big)
    sub.field(name="big_sharded", path="source/big_sharded", kind="map", manifold_id=sid, data=big,
              meta=array_meta(np.float32, (nv, 1000), chunks=(256, 500), shards=(1024, 1000)))
    sm = np.arange(300 * 200, dtype=np.float64) * 0.5
    sm[np.arange(300 * 200) % 61 == 0] = 0.0
    sm = sm.reshape(300, 200)
    sub.field(name="small_sharded", path="source/small_sharded", kind="map", manifold_id=sid, data=sm,
              meta=array_meta(np.float64, (300, 200), chunks=(100, 100), shards=(300, 200)))
    sub.field(name="dense_zeros", path="source/dense_zeros", kind="map", manifold_id=sid, value_type="integer", data=np.zeros(50, np.uint8), dense=True)
    sub.field(name="sparse", path="source/sparse", kind="map", manifold_id=sid,
              sparse={"data": np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], np.float32), "indices": np.array([0, 3, 1, 0, 2, 4, 3], np.int32),
                      "indptr": np.array([0, 2, 3, 3, 6, 7], np.int32)})
    codes = ((np.arange(nv) // 100) % 5).astype(np.int32)
    did, _ = sub.dictionary("parcels", ["alpha", "beta", "gamma", "delta", "epsilon"])
    sub.selection(name="parcels", path="source/parcels", type="set", manifold_id=sid, data=codes, dictionary_id=did, cell="0",
                  measurements=[{"code": 1, "of_field_id": small_id, "measure": "mean", "value": 12.5, "unit": "a.u."}])
    sub.operator(name="affine", path="affine", kind="transition", from_manifold_id=sid, n_rows=3, n_cols=4, data=np.arange(12, dtype=np.float64).reshape(3, 4) * 0.5)
    sub.complete()
    again = SubjectWriter.open(ds, "S1", created_by="parity")
    again.field(name="extra", path="source/extra", kind="map", manifold_id=sid, data=small)
    again.complete()
    ds.close()


def interrupt(folder: Path, subject: str) -> None:
    from nxr.io.writer import SubjectWriter, open_writer
    deterministic(1000)
    ds = open_writer(folder)
    sub = SubjectWriter.open(ds, subject, created_by="parity")
    sid = sub.find("manifold", "source/sphere")["id"]
    sub.field(name="interrupted", path="source/interrupted", kind="map", manifold_id=sid, data=ramp(100))
    ds.close()          # never completed


# ── the dump: every row of every table, every file ──────────────────────────────────────────────────────────────────

SKIP_COLUMNS = {("sync_job", "done_utc"), ("sync_job", "created_utc")}


def dump_rows(dataset: Path, datastore_url: str | None = None) -> dict:
    con = sqlite3.connect(str(dataset / "dataset.sqlite"))
    out = {}
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for t in tables:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        rows = []
        for r in con.execute(f"SELECT * FROM {t}"):
            d = {c: v for c, v in zip(cols, r) if (t, c) not in SKIP_COLUMNS}
            if t == "datastore":
                d["url"] = "<datastore>"
            rows.append(d)
        rows.sort(key=lambda d: json.dumps(d, sort_keys=True, default=str))
        out[t] = rows
    out["_user_version"] = con.execute("PRAGMA user_version").fetchone()[0]
    con.close()
    return out


def small_files():
    """nxr's one helper for a dataset's keys (format 52: rows of each subject's small_files.sqlite; format 51: files)"""
    from nxr.io import small_files as sf
    return sf


def subject_keys(subject: Path) -> list[str]:
    """every key of a subject's folder, rows and files merged (the database file itself is not a key)"""
    loc = small_files().locate(subject)
    if loc is not None:
        return loc[0].keys(loc[1])
    return sorted(p.relative_to(subject).as_posix() for p in subject.rglob("*") if p.is_file())


def read_key(subject: Path, key: str) -> bytes:
    sf = small_files()
    loc = sf.locate(subject / key)
    if loc is not None and loc[1]:
        data = loc[0].get_bytes(loc[1])
        assert data is not None, key
        return data
    return (subject / key).read_bytes()


def array_docs(subject: Path) -> dict:
    """every zarr.json of a subject, parsed: key (the node's path) -> document"""
    return {k[: -len("/zarr.json")] if "/" in k else "": json.loads(read_key(subject, k)) for k in subject_keys(subject) if k.split("/")[-1] == "zarr.json"}


def dump_files(root: Path) -> dict:
    """KEY -> ('text', text) for zarr.json and the catalog, ('bytes', sha256) for chunks. At format 52 the keys of a subject's small_files.sqlite are listed too,
    each with a '[row] ' prefix so that the place a key lives in is compared as well; the lock file and the databases themselves are not keys."""
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith(".writer.lock"):
            continue
        rel = p.relative_to(root).as_posix()
        if rel.endswith(".sqlite") or "dataset.sqlite" in rel or "small_files.sqlite" in rel:
            continue
        out[rel] = _entry(p.name, p.read_bytes())
    for db in sorted(root.rglob("small_files.sqlite")):
        con = sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True)
        for key, value in con.execute("SELECT key, value FROM small_files ORDER BY key"):
            out["[row] " + db.parent.relative_to(root).as_posix() + "/" + key] = _entry(key.split("/")[-1], bytes(value))
        con.close()
    return out


def _entry(name: str, data: bytes):
    if name in ("zarr.json", "catalog.json"):
        t = data.decode("utf-8")
        if name == "catalog.json":        # rewritten at every change, stamped with the clock
            t = re.sub(r'"modified_utc": "[^"]*"', '"modified_utc": "<T>"', t)
        return ("text", t)
    return ("bytes", hashlib.sha256(data).hexdigest())
