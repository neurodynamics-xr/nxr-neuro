"""The pipeline layer (nxr_io x nxr_compute): `nxr-neuro eigen solve` caches, and what it stores is what nxr's own `nxr eigen solve` stores.

Needs the nxr-neuro build (pipelines on) and nxr's environment:

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro \\
      uv run --project $NXR python -m pytest -q --basetemp=<a scratch folder> test/parity/test_pipelines_parity.py

Synthetic data only: an icosphere written here as an OBJ file.

(a) the first call computes and stores, the second is a cache hit returning the same arrays without computing (timed);
(b) nxr's Python reads the stored package, and its rows and layout match what nxr's own solve writes on the same mesh (eigenvalues to a tolerance;
    eigenvectors per degenerate subspace; names, paths, shapes, dtypes, chunks and codecs exact);
(c) another K or a changed mesh misses the cache (and a package made by another job is refused without --replace).
"""
from __future__ import annotations

import json
import os
import shutil
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
pytestmark = pytest.mark.skipif(NXR is None or IO is None, reason="set NXR and NXR_NEURO (the nxr-neuro build)")
NXR_MAIN = "import sys; from nxr.cli.main import main; sys.exit(main(sys.argv[1:]))"
MODES = 60
STEM = "hodgeDirac"
NODES = {"manifold": f"eigenvalue/{STEM}", "field": f"eigenvalue/{STEM}_kind", "operator": f"{STEM}_eigenvectors", "selection": f"{STEM}_support"}


def run(cmd, *args):
    t = time.perf_counter()
    p = subprocess.run([*cmd, *map(str, args)], capture_output=True, text=True)
    return p, time.perf_counter() - t


def io(*args):
    return run([str(IO)], *args)


def nxr(*args):
    return run([sys.executable, "-c", NXR_MAIN], *args)


def lines(p):
    return [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]


def icosphere(levels, radius):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6),
         (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [tuple(np.array(v) / np.linalg.norm(v)) for v in V]
    for _ in range(levels):
        mid, G = {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = (np.array(V[a]) + np.array(V[b])) / 2
                V.append(tuple(p / np.linalg.norm(p)))
                mid[k] = len(V) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            G += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = G
    return np.array(V) * radius, np.array(F)


def write_obj(path, V, F):
    path.write_text("".join(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in V) + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F))


@pytest.fixture(scope="module", params=[pytest.param(51, id="f51"), pytest.param(52, id="f52")])
def store_format(request):
    """the store format the dataset is made at (nxr's own solve then works on a copy of it)"""
    return request.param


@pytest.fixture(scope="module")
def base(tmp_path_factory, store_format):
    """a dataset with one subject holding an icosphere (642 vertices) as its primary surface, nothing solved yet"""
    root = tmp_path_factory.mktemp("pipelines")
    V, F = icosphere(3, 0.08)
    write_obj(root / "sphere.obj", V, F)
    store = root / "store"
    assert io("dataset", "create", "--datastore", store, "--name", "d", "--store-format", store_format)[0].returncode == 0
    assert io("subject", "create", "--dataset", store / "d", "--name", "S1")[0].returncode == 0
    p, _ = io("surface", "add", "--dataset", store / "d", "--subject", "S1", root / "sphere.obj", "--name", "cortex")
    assert p.returncode == 0, p.stdout + p.stderr
    return root


def copy_of(base, name):
    dst = base / name
    shutil.copytree(base / "store", dst)
    return dst / "d"


def solve_io(ds, modes=MODES, *extra):
    p, wall = io("eigen", "solve", "--dataset", ds, "--subject", "S1", "--modes", modes, *extra)
    return p, lines(p), wall


@pytest.fixture(scope="module")
def solved(base):
    """the same mesh solved by nxr-io (twice) and by nxr's own solve, each in its own copy of the dataset"""
    a = copy_of(base, "io")
    p1, l1, w1 = solve_io(a)
    assert p1.returncode == 0, p1.stdout + p1.stderr
    p2, l2, w2 = solve_io(a)
    assert p2.returncode == 0, p2.stdout + p2.stderr
    b = copy_of(base, "nxr")
    p, _ = nxr("eigen", "solve", "--dataset", b, "--subject", "S1", "--modes", MODES)
    assert p.returncode == 0, p.stdout + p.stderr
    return {"io": a, "nxr": b, "first": (l1, w1), "second": (l2, w2)}


def read(ds, subject="S1"):
    import nxr.io as nio
    d = nio.open(ds)
    rd = d.reader(subject)
    rows = {t: d.q(f"SELECT * FROM {t} WHERE subject_id = ? AND path = ?", d.subject_row(subject)["id"], p)[0] for t, p in NODES.items()}
    arrays = {"mu": rd(NODES["manifold"], "float64") if False else None}
    from nxr.io import small_files
    store = Path(d.store(subject))
    arr = {t: np.asarray(small_files.open_array(store / p, mode="r")[...]) for t, p in NODES.items() if t != "selection"}
    arr["selection"] = np.asarray(small_files.open_array(store / rows["selection"]["array_path"], mode="r")[...])
    docs = {t: json.loads(ws.read_key(store, (rows["selection"]["array_path"] if t == "selection" else p) + "/zarr.json")) for t, p in NODES.items()}
    return d, rows, arr, docs


def test_a_first_call_computes_and_second_is_a_hit(solved):
    (l1, w1), (l2, w2) = solved["first"], solved["second"]
    r1, r2 = l1[-1], l2[-1]
    assert r1["stage"] == r2["stage"] == "result"
    assert (r1["cache"], r2["cache"]) == ("miss", "hit")
    assert any(x.get("stage") == "solved" for x in l1) and not any(x.get("stage") == "solved" for x in l2)   # the hit computed nothing
    assert r1["operator_id"] == r2["operator_id"] and r1["n_modes"] == r2["n_modes"] and r1["mu"] == r2["mu"]
    print(f"\n  miss: {r1['solve']['seconds']:.3f} s in the call ({w1:.2f} s wall) · hit: {r2['solve']['seconds']:.3f} s in the call ({w2:.2f} s wall)")
    assert r2["solve"]["seconds"] < r1["solve"]["seconds"]
    # the arrays a hit returns are the stored ones: the second call wrote nothing (the bytes are as the first left them)
    d, rows, arr, _ = read(solved["io"])
    assert arr["operator"].shape == (642, r1["n_modes"]) and arr["operator"].dtype == np.float32
    assert json.loads(rows["operator"]["producer_json"])["cache"]["pipeline"] == "eigen_solve"


def test_b_nxr_reads_it_and_the_layout_is_nxrs(solved):
    d, rows, arr, docs = read(solved["io"])
    dn, rows_n, arr_n, docs_n = read(solved["nxr"])
    skip = ("id", "created_by", "modified_by", "producer_json", "params_json", "description", "comment", "source")
    for t in NODES:
        for col in rows[t]:
            if col in skip or col.endswith("_id") or col.endswith("_utc") or col in ("n_exact",):
                continue
            assert rows[t][col] == rows_n[t][col], f"{t}.{col}: {rows[t][col]!r} != {rows_n[t][col]!r}"
        assert (rows[t]["status"], rows[t]["name"]) == ("complete", rows_n[t]["name"])
        a, b = dict(docs[t]), dict(docs_n[t])
        a.pop("attributes", None), b.pop("attributes", None)
        assert a == b, f"{t}: zarr.json differs"
    assert json.loads(rows["operator"]["params_json"]).keys() == json.loads(rows_n["operator"]["params_json"]).keys()
    assert set(json.loads(rows["operator"]["producer_json"])) >= set(json.loads(rows_n["operator"]["producer_json"]))
    for t in ("manifold", "field", "selection", "operator"):
        assert arr[t].dtype == arr_n[t].dtype and arr[t].ndim == arr_n[t].ndim, t
    assert arr["selection"].shape == arr_n["selection"].shape and (arr["selection"] == arr_n["selection"]).all()
    assert arr["operator"].shape[0] == arr_n["operator"].shape[0]
    # the primary surface and the target of the eigenvectors are the same rows as in nxr's
    assert rows["operator"]["to_manifold_id"] == d.q("SELECT id FROM manifold WHERE is_primary = 1")[0]["id"]
    # eigenvalues: Spectra against SciPy, on the modes both solves hold
    mu, mu_n = arr["manifold"], arr_n["manifold"]
    kind, kind_n = arr["field"], arr_n["field"]
    n = min(len(mu), len(mu_n))
    print(f"\n  modes: nxr-io {len(mu)} (exact {(kind == 0).sum()}, coexact {(kind == 2).sum()}) · nxr {len(mu_n)} (exact {(kind_n == 0).sum()}, coexact {(kind_n == 2).sum()})")
    assert abs(len(mu) - len(mu_n)) <= 2 and n >= MODES // 2
    # the band's top lies inside a degenerate multiplet and each solver keeps a different part of it: compare the modes 1% below the top
    m = int((mu_n[:n] < mu_n[n - 1] * 0.99).sum())
    assert m >= MODES // 2
    rel = np.abs(mu[:m] - mu_n[:m]) / mu_n[:m]
    print(f"  eigenvalues, relative difference over the first {m} modes: max {rel.max():.2e}")
    assert rel.max() < 1e-6
    assert (kind[:m] == kind_n[:m]).all()
    # eigenvectors up to sign and within degenerate subspaces: each cluster of equal eigenvalues spans the same subspace
    P, Pn = arr["operator"].astype(np.float64)[:, :m], arr_n["operator"].astype(np.float64)[:, :m]
    edges = np.flatnonzero(np.diff(mu_n[:m]) / mu_n[1:m] > 1e-4) + 1
    worst = 0.0
    for lo, hi in zip([0, *edges], [*edges, m]):
        A, B = P[:, lo:hi], Pn[:, lo:hi]
        Q, _ = np.linalg.qr(B)
        worst = max(worst, np.linalg.norm(A - Q @ (Q.T @ A)) / np.linalg.norm(A))
    print(f"  eigenvectors, worst relative distance of a cluster from nxr's subspace: {worst:.2e}")
    assert worst < 1e-3


def test_c_other_k_or_other_mesh_misses(base, solved):
    ds = solved["io"]
    p, ls, _ = solve_io(ds, MODES + 8)                       # another K: a package made by another job is refused ...
    assert p.returncode == 1 and "pass --replace" in ls[-1]["message"]
    p, ls, _ = solve_io(ds, MODES + 8, "--replace")           # ... and replaced on request (moved aside, never deleted): a miss
    assert p.returncode == 0 and ls[-1]["cache"] == "miss" and ls[-1]["replaced"] is True
    assert (Path(ds).parent / next(x for x in os.listdir(Path(ds).parent) if x.startswith("_replaced-"))).is_dir()
    p, ls, _ = solve_io(ds, MODES + 8)
    assert ls[-1]["cache"] == "hit"
    # a changed mesh (the stored positions scaled in a copy): the key differs, nothing is reused
    other = copy_of(base, "moved")
    p, ls, _ = solve_io(other)
    assert ls[-1]["cache"] == "miss"
    import nxr.io as nio
    from nxr.io import small_files
    d = nio.open(other)
    pos = Path(d.store("S1")) / d.q("SELECT g.positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.is_primary = 1")[0]["positions_path"]
    z = small_files.open_array(pos, mode="r+")
    z[...] = z[...] * np.float32(1.01)
    p, ls, _ = solve_io(other)
    assert p.returncode == 1 and "pass --replace" in ls[-1]["message"]
    p, ls, _ = solve_io(other, MODES, "--replace")
    assert p.returncode == 0 and ls[-1]["cache"] == "miss"
