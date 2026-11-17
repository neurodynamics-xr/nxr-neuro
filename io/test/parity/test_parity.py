"""Parity: `nxr-neuro read` (C++) decodes every array of an nxr dataset to the same bytes nxr's Python (zarr) does.

Run with nxr's environment (it has zarr and the writer), pointing at nxr's checkout and the built CLI:

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro \\
      uv run --project $NXR python -m pytest -q test/parity

Datasets are synthetic only: nxr's fixture (tests/io/fixtures/v50_atlas.tar.gz), and one GENERATED here with nxr's own writer
(an icosphere, big arrays so `array_meta` blosc-compresses and shards them, plus raw zarr arrays for gzip / zstd / crc32c /
big-endian / complex / missing chunks).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent))
import writer_scenario as ws  # noqa: E402

NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
CLI = Path(os.environ["NXR_NEURO"]) if "NXR_NEURO" in os.environ else None
pytestmark = pytest.mark.skipif(NXR is None or CLI is None, reason="set NXR (nxr checkout) and NXR_NEURO (the built nxr-neuro)")


def run(*args, ok=True):
    p = subprocess.run([str(CLI), "read", *map(str, args)], capture_output=True)
    if ok:
        assert p.returncode == 0, p.stderr.decode()
    return p


def crc(b: bytes) -> str:
    return f"{zlib.crc32(b) & 0xFFFFFFFF:08x}"


def raw(a: np.ndarray) -> bytes:
    """What the C++ returns: C order, little-endian."""
    return np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes()


def arrays_of(subject_store: Path):
    """Every zarr array under a subject's store: (store-relative path, zarr array)."""
    from nxr.io import small_files
    for rel, doc in sorted(ws.array_docs(subject_store).items()):
        if doc["node_type"] == "array":
            yield rel, small_files.open_array(subject_store / rel, mode="r")


def subjects(dataset: Path):
    return [(line.split("\t")[1], dataset / line.split("\t")[2]) for line in run(dataset).stdout.decode().splitlines() if line.startswith("subject\t")]


def check_dataset(dataset: Path, seed: int = 0) -> dict:
    """Whole arrays, the arrays the rows name, and random regions; returns what was exercised."""
    rng = np.random.default_rng(seed)
    seen = {"arrays": 0, "regions": 0, "codecs": set(), "sharded": 0, "dtypes": set()}
    listed = {}
    for line in run(dataset).stdout.decode().splitlines():
        f = line.split("\t")
        if f[0] != "subject":
            listed[(f[0], f[2])] = (f[3], f[4], f[5])
    named = 0
    for name, store in subjects(dataset):
        for rel, a in arrays_of(store):
            want = a[...] if a.shape else a[()]
            want = np.asarray(want)
            out = dataset / f"_out.bin"
            p = run(dataset, "--array", name, rel, "--out", out)
            shape, dtype, c = p.stdout.decode().strip().split("\t")
            assert json.loads(shape) == list(a.shape), (name, rel)
            assert dtype == a.dtype.name, (name, rel)
            got = out.read_bytes()
            assert got == raw(want), f"{name}:{rel} differs"
            assert c == crc(raw(want))
            seen["arrays"] += 1
            seen["dtypes"].add(a.dtype.name)
            codecs = json.loads(ws.read_key(store, rel + "/zarr.json"))["codecs"]
            for k in codecs:
                seen["codecs"].add(k["name"])
                if k["name"] == "sharding_indexed":
                    seen["sharded"] += 1
                    seen["codecs"].update(i["name"] for i in k["configuration"]["codecs"] + k["configuration"]["index_codecs"])
            if (name, rel) in listed:
                named += 1
                assert listed[(name, rel)] == (shape.replace(" ", ""), dtype, c), (name, rel)
            if a.size and a.ndim:
                for _ in range(3):
                    lo = [int(rng.integers(0, s)) for s in a.shape]
                    hi = [int(rng.integers(l + 1, s + 1)) for l, s in zip(lo, a.shape)]
                    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
                    reg = ",".join(f"{l}:{h}" for l, h in zip(lo, hi))
                    run(dataset, "--array", name, rel, "--region", reg, "--out", out)
                    assert out.read_bytes() == raw(a[sl]), f"{name}:{rel}[{reg}] differs"
                    seen["regions"] += 1
            out.unlink(missing_ok=True)
    assert named == len(listed) > 0, "every array a row names is listed, and was compared"
    seen["named"] = named
    return seen


# ── nxr's fixture ──────────────────────────────────────────────────────────────────────────────────
def extract(name: str, into: Path) -> Path:
    with tarfile.open(NXR / "tests" / "io" / "fixtures" / name) as t:
        t.extractall(into, filter="data")
    return into / "ds"


def test_fixture_v50(tmp_path):
    seen = check_dataset(extract("v50_atlas.tar.gz", tmp_path))
    print("fixture:", seen)
    assert seen["sharded"] > 0 and seen["arrays"] > 20


def test_refuses_version_49(tmp_path):
    ds = extract("v49_synthetic.tar.gz", tmp_path)
    p = run(ds, ok=False)
    assert p.returncode != 0
    msg = p.stderr.decode()
    assert "user_version 49, this build reads 51" in msg
    assert "run `nxr migrate --dataset <dataset folder>`" in msg


def test_refuses_other_version(tmp_path):
    import sqlite3
    ds = extract("v50_atlas.tar.gz", tmp_path)
    c = sqlite3.connect(ds / "dataset.sqlite")
    c.execute("PRAGMA user_version = 31")
    c.close()
    msg = run(ds, ok=False).stderr.decode()
    assert "user_version 31, this build reads 51" in msg and "re-convert it from its source" in msg


def test_no_database(tmp_path):
    assert "no dataset database here" in run(tmp_path, ok=False).stderr.decode()


# ── a dataset generated with nxr's own writer ──────────────────────────────────────────────────────
def icosphere(levels: int):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6),
         (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [tuple(np.array(p) / np.linalg.norm(p)) for p in v]
    for _ in range(levels):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = np.array(v[a]) + np.array(v[b])
                v.append(tuple(m / np.linalg.norm(m)))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    from nxr.cli.main import main
    from nxr.io import writer
    from nxr.io.writer import SubjectWriter, array_meta
    root = tmp_path_factory.mktemp("generated")
    store = root / "store"
    V, F = icosphere(4)
    obj = root / "sphere.obj"
    obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in V) + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F))
    ds = store / "gen"
    for argv in (["dataset", "create", "--datastore", store, "--name", "gen"],
                 ["subject", "create", "--dataset", ds, "--name", "sub-01"],
                 ["surface", "add", "--dataset", ds, "--subject", "sub-01", obj, "--name", "sphere"]):
        assert main([str(a) for a in argv]) == 0
    rng = np.random.default_rng(7)
    with writer.open_writer(ds) as w:
        sub = SubjectWriter.open(w, "sub-01", immediate=True)
        mid = sub.db.one("SELECT id FROM manifold WHERE is_primary = 1")["id"]
        cases = {   # name -> (array, array_meta kwargs): how nxr itself lays each out
            "blosc_float32": (rng.standard_normal((45000, 32), dtype=np.float32), {}),                       # > 4 MiB: blosc, ~2 MiB chunks
            "blosc_sharded_float32": (rng.standard_normal((45000, 32), dtype=np.float32), {"chunks": (1000, 32), "shards": (7000, 32), "compress": True}),
            "blosc_float64": (rng.standard_normal((1300, 70, 7)), {"compress": True, "chunks": (300, 70, 7)}),
            "blosc_int32": (rng.integers(-5, 5, (1_200_000,), dtype=np.int32), {}),
            "blosc_uint8": (rng.integers(0, 4, (3, 1_500_000), dtype=np.uint8), {"compress": True}),
            "blosc_int64": (rng.integers(-9, 9, (700_000,), dtype=np.int64), {}),
            "plain_sharded_float32": (rng.standard_normal((2000, 33), dtype=np.float32), {"chunks": (1, 33), "shards": (64, 33), "compress": False}),
            "single_chunk": (rng.standard_normal((50, 3), dtype=np.float32), {}),
            "uint32": (rng.integers(0, 9, (11, 5), dtype=np.uint32), {}),
        }
        for k, (data, kw) in cases.items():
            sub.field(name=k, path=f"gen/{k}", kind="scalar map", manifold_id=mid, meta=array_meta(data.dtype, data.shape, **kw), data=data)
        sub.complete()
    sroot = ds / "sub-01.nxr.zarr"
    # raw zarr arrays NOT laid out by array_meta: codecs nxr's reader accepts but its writer does not choose
    from zarr.codecs import BloscCodec, BytesCodec, GzipCodec, ShardingCodec, ZstdCodec
    x = rng.standard_normal((300, 50)).astype(np.float32)
    raws = {
        "raw/gzip": dict(compressors=[GzipCodec(level=4)]),
        "raw/zstd": dict(compressors=[ZstdCodec(level=3, checksum=True)]),
        "raw/zstd_crc": dict(compressors=[ZstdCodec(level=1)]),
        "raw/big_endian": dict(serializer=BytesCodec(endian="big"), compressors=None),
        "raw/blosc_lz4_bitshuffle": dict(compressors=[BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")]),
        "raw/shard_start_index": dict(chunks=(120, 50), compressors=None, serializer=ShardingCodec(
            chunk_shape=(40, 25), codecs=[BytesCodec(), GzipCodec(level=1)], index_location="start")),
    }
    for path, kw in raws.items():
        kw.setdefault("chunks", (128, 50))
        a = zarr.create_array(str(sroot / path), shape=x.shape, dtype=x.dtype, **kw)
        a[...] = x
    z = zarr.create_array(str(sroot / "raw/complex"), shape=(10, 7), dtype="complex64", chunks=(4, 4))
    z[...] = (rng.standard_normal((10, 7)) + 1j * rng.standard_normal((10, 7))).astype(np.complex64)
    z = zarr.create_array(str(sroot / "raw/missing_chunks"), shape=(100, 10), dtype="int32", chunks=(10, 10), fill_value=-7,
                          config={"write_empty_chunks": False})
    z[0:10] = 3
    z[55:60] = -7   # equals the fill value: not written
    z = zarr.create_array(str(sroot / "raw/scalar"), shape=(), dtype="float64", chunks=())
    z[()] = 2.5
    z = zarr.create_array(str(sroot / "raw/empty"), shape=(0, 4), dtype="float32", chunks=(1, 4))
    return ds


def test_generated_dataset(generated):
    seen = check_dataset(generated)
    print("generated:", seen)
    assert {"blosc", "sharding_indexed", "crc32c", "gzip", "zstd", "bytes"} <= seen["codecs"]
    assert {"float32", "float64", "int32", "int64", "uint8", "uint32", "complex64"} <= seen["dtypes"]
    assert seen["named"] >= 9 and seen["sharded"] >= 3
    assert '"index_location":"start"' in ws.read_key(generated / "sub-01.nxr.zarr", "raw/shard_start_index/zarr.json").decode().replace(" ", "")
    # blosc really is in play for the big arrays nxr laid out
    meta = json.loads(ws.read_key(generated / "sub-01.nxr.zarr", "gen/blosc_float32/zarr.json"))
    inner = lambda d: d["codecs"][0]["configuration"]["codecs"] if d["codecs"][0]["name"] == "sharding_indexed" else d["codecs"]     # an array over 4 MiB is sharded: its blosc is the inner chunks
    assert any(c["name"] == "blosc" for c in inner(meta))
    sharded = json.loads(ws.read_key(generated / "sub-01.nxr.zarr", "gen/blosc_sharded_float32/zarr.json"))["codecs"][0]
    assert sharded["name"] == "sharding_indexed" and any(c["name"] == "blosc" for c in sharded["configuration"]["codecs"])


def test_regions_across_chunk_and_shard_edges(generated):
    """Slices that start and end inside different chunks and shards (chunk 1000 rows, shard 7000 rows)."""
    from nxr.io import small_files
    a = small_files.open_array(generated / "sub-01.nxr.zarr/gen/blosc_sharded_float32", mode="r")
    for lo, hi in [(999, 1001), (6999, 7001), (6500, 14100), (44990, 45000), (0, 45000), (13999, 14000)]:
        for c0, c1 in [(0, 32), (5, 6), (30, 32)]:
            out = generated / "_r.bin"
            run(generated, "--array", "sub-01", "gen/blosc_sharded_float32", "--region", f"{lo}:{hi},{c0}:{c1}", "--out", out)
            assert out.read_bytes() == raw(a[lo:hi, c0:c1])
    b = small_files.open_array(generated / "sub-01.nxr.zarr/gen/blosc_float32", mode="r")
    meta = json.loads(ws.read_key(generated / "sub-01.nxr.zarr", "gen/blosc_float32/zarr.json"))
    rows = (meta["codecs"][0]["configuration"] if meta["codecs"][0]["name"] == "sharding_indexed" else meta["chunk_grid"]["configuration"])["chunk_shape"][0]
    out = generated / "_r.bin"
    run(generated, "--array", "sub-01", "gen/blosc_float32", "--region", f"{rows - 2}:{rows + 2},3:9", "--out", out)
    assert out.read_bytes() == raw(b[rows - 2:rows + 2, 3:9])
    out.unlink()
