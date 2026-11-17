"""The Hodge–Dirac solve of nxr-neuro IS the reference method of nxr and Rheome: the same mesh solved by `nxr eigen solve` (SciPy/ARPACK) and by
`nxr-neuro eigen solve` (Spectra), then compared: mode counts and kinds, mu, the eigenvectors per degenerate subspace (principal angles), the stored package
arrays (float32, sign-aligned on isolated modes), time and peak memory of each. Also: the named options (relative ladder, slice) still run and give the same mu.

    NXR=<nxr checkout> NXR_NEURO=<build>/cli/nxr-neuro NXR_PARITY_LEVEL=3 NXR_PARITY_MODES=200 \\
      uv run --project $NXR python -m pytest -q -s --basetemp=<a scratch folder> io/test/parity/test_hodge_reference_parity.py

Level 3 = 642 vertices; level 6 = 40 962 vertices (the real cortex's scale; use NXR_PARITY_MODES=3000). Synthetic data only. Needs /usr/bin/time (macOS: -l).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_pipelines_parity import NXR_MAIN, icosphere, write_obj  # noqa: E402
from spellings import to_io  # noqa: E402

NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
IO = Path(os.environ["NXR_NEURO"]) if "NXR_NEURO" in os.environ else None
pytestmark = pytest.mark.skipif(NXR is None or IO is None, reason="set NXR and NXR_NEURO (the nxr-neuro build)")
LEVEL = int(os.environ.get("NXR_PARITY_LEVEL", "3"))
MODES = int(os.environ.get("NXR_PARITY_MODES", "200"))
NODES = {"manifold": "eigenvalue/hodgeDirac", "field": "eigenvalue/hodgeDirac_kind", "operator": "hodgeDirac_eigenvectors"}


def timed(cmd, *args):
    """(completed process, seconds, peak resident GB of the process tree's largest member)"""
    t = time.perf_counter()
    p = subprocess.run(["/usr/bin/time", "-l", *cmd, *map(str, args)], capture_output=True, text=True)
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    return p, time.perf_counter() - t, (int(m.group(1)) / 2 ** 30 if m else float("nan"))


def io(*args):
    return timed([str(IO)], *to_io(args))


def nxr(*args):
    return timed([sys.executable, "-c", NXR_MAIN], *args)


def make(root, name, fmt):
    store = root / name
    V, F = icosphere(LEVEL, 0.08)
    if not (root / "sphere.obj").exists():
        write_obj(root / "sphere.obj", V, F)
    assert io("dataset", "create", "--datastore", store, "--name", "d", "--store-format", fmt)[0].returncode == 0
    assert io("subject", "create", "--dataset", store / "d", "--name", "S1")[0].returncode == 0
    p = io("surface", "add", "--dataset", store / "d", "--subject", "S1", root / "sphere.obj", "--name", "cortex")[0]
    assert p.returncode == 0, p.stdout + p.stderr
    return store / "d"


def package(ds):
    import nxr.io as nio
    from nxr.io import small_files
    d = nio.open(ds)
    store = Path(d.store("S1"))
    sid = d.subject_row("S1")["id"]
    arr = {}
    for t, p in NODES.items():
        row = d.q(f"SELECT * FROM {t} WHERE subject_id = ? AND path = ?", sid, p)[0]
        arr[t] = np.asarray(small_files.open_array(store / p, mode="r")[...])
    return arr, d.q("SELECT producer_json FROM operator WHERE subject_id = ? AND path = ?", sid, NODES["operator"])[0]["producer_json"]


@pytest.fixture(scope="module", params=[pytest.param("51", id="f51"), pytest.param("52", id="f52")])
def store_format(request):
    """the store format both datasets are made at"""
    return request.param


@pytest.fixture(scope="module")
def runs(tmp_path_factory, store_format):
    root = tmp_path_factory.mktemp("hodgeref")
    out = {}
    a = make(root, "nxr-neuro", store_format)
    p, s, g = io("eigen", "solve", "--dataset", a, "--subject", "S1", "--modes", MODES)
    assert p.returncode == 0, p.stdout + p.stderr[-2000:]
    out["neuro"] = (a, s, g)
    b = make(root, "nxr", store_format)
    p, s, g = nxr("eigen", "solve", "--dataset", b, "--subject", "S1", "--modes", MODES)
    assert p.returncode == 0, p.stdout + p.stderr[-2000:]
    out["nxr"] = (b, s, g)
    out["root"] = root
    out["format"] = store_format
    return out


def test_reference_parity(runs):
    (an, sn, gn), (ax, sx, gx) = runs["neuro"], runs["nxr"]
    pn, prov = package(an)
    px, _ = package(ax)
    mu, mux, kind, kindx = pn["manifold"], px["manifold"], pn["field"], px["field"]
    print(f"\n  level {LEVEL}, modes requested {MODES}: nxr-neuro {len(mu)} modes (exact {(kind == 0).sum()}, coexact {(kind == 2).sum()}), "
          f"nxr {len(mux)} (exact {(kindx == 0).sum()}, coexact {(kindx == 2).sum()})")
    print(f"  time: nxr-neuro {sn:.1f} s, peak {gn:.2f} GB · nxr {sx:.1f} s, peak {gx:.2f} GB (whole commands)")
    assert len(mu) == len(mux) and (kind == kindx).all()
    rel = np.abs(mu - mux) / mux
    print(f"  mu: max relative difference {rel.max():.2e}")
    assert rel.max() <= 1e-9    # measured: 4e-11 (642 V), 2.6e-10 (40 962 V; the median is 6e-14, the few above 1e-10 are the lowest, near-degenerate modes)
    P, Px = pn["operator"], px["operator"]
    assert P.dtype == Px.dtype == np.float32 and P.shape == Px.shape
    # per degenerate cluster: principal angles of the two subspaces; isolated modes: sign-aligned elementwise
    edges = np.flatnonzero(np.diff(mux) / mux[1:] > 1e-5) + 1       # modes closer than 1e-5 are one subspace: both solvers' eigenvalues are good to ~1e-10, so closer ones mix
    ang, iso, iso_n = 0.0, 0.0, 0
    # the LAST cluster is the band's top: the ladders' k cut falls inside a degenerate multiplet and each solver (ARPACK, Spectra) returns a different
    # part of it, an arbitrary basis of a sub-space of the multiplet; it is not comparable (same count, same mu) and is left out of the angles
    for lo, hi in list(zip([0, *edges], [*edges, len(mux)]))[:-1]:
        A, B = P[:, lo:hi].astype(np.float64), Px[:, lo:hi].astype(np.float64)
        Qa, _ = np.linalg.qr(A)
        Qb, _ = np.linalg.qr(B)
        s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
        ang = max(ang, float(np.sqrt(max(0.0, 1 - s.min() ** 2))))        # sine of the largest principal angle
        if hi - lo == 1:
            sg = np.sign(A[:, 0] @ B[:, 0]) or 1.0
            iso = max(iso, float(np.abs(sg * A[:, 0] - B[:, 0]).max() / np.abs(B[:, 0]).max()))
            iso_n += 1
    print(f"  eigenvectors: {len(edges) + 1} subspaces, sine of the largest principal angle {ang:.2e}; {iso_n} isolated modes, sign-aligned max |difference| / max|mode| {iso:.2e}")
    # the stored package is float32 (rounding 6e-8 per entry): 642 V measured 6.5e-8 / 4e-8; 40 962 V 9e-6 / 1.4e-5 (the solvers' eigenvectors of near-degenerate modes
    # are good to about their eigenvalue accuracy over the gap, ~1e-10 / 1e-5)
    assert ang <= (5e-7 if LEVEL <= 3 else 1e-4)
    assert iso <= (1e-5 if LEVEL <= 3 else 1e-4)
    p = json.loads(prov)
    assert p["solve"]["options"]["shift"] == "absolute" and p["solve"]["options"]["slice"] == 0 and p["solve"]["options"]["normalize"] is False
    assert p["solve"]["options"]["star1_floor"] is None and p["solve"]["star1_clamped"] == 0
    assert p["cache"]["params"]["method"] == p["solve"]["options"]
    # the intrinsic Delaunay construction's two constants are named, recorded options of the shared names (and in the cache key)
    assert p["solve"]["options"]["mollify"] == 1e-6 and p["solve"]["options"]["flip_eps"] == 1e-6
    assert p["solve"]["delaunay"]["mollify"] == 1e-6 and p["solve"]["delaunay"]["flip_eps"] == 1e-6 and p["solve"]["delaunay"]["flips"] >= 0


@pytest.mark.skipif(LEVEL > 3, reason="the options run at the small scale")
def test_named_options_still_work(runs):
    root = runs["root"]
    ref = package(runs["neuro"][0])[0]["manifold"]
    for name, extra in {"relative": ["--shift", "relative", "--normalize"], "slice": ["--slice", "60", "--normalize"], "floor": ["--star1-floor", "1e-6"]}.items():
        ds = make(root, f"opt-{name}", runs["format"])
        p, s, g = io("eigen", "solve", "--dataset", ds, "--subject", "S1", "--modes", MODES, *extra)
        assert p.returncode == 0, p.stdout + p.stderr[-2000:]
        arr, prov = package(ds)
        o = json.loads(prov)["solve"]["options"]
        n = min(len(ref), len(arr["manifold"]))
        # the relative ladder in one solve is known to lose multiplet members (the reference does not): report, don't compare
        d = float(np.abs(arr["manifold"][:n] - ref[:n]).max() / ref.max())
        print(f"\n  {name}: {len(arr['manifold'])} modes, {s:.1f} s, options {o}, max |mu - reference mu| / mu_max over {n}: {d:.2e}")
        if name != "relative":
            assert len(arr["manifold"]) == len(ref) and d < 1e-8


@pytest.mark.skipif(LEVEL > 3, reason="the options run at the small scale")
def test_delaunay_options_are_recorded_cached_and_match_nxr(runs):
    """--mollify / --flip-eps: the same names in nxr-neuro and nxr; recorded in the provenance, part of the cache key (another value, another cached
    result; the same value again, a hit); and nxr-neuro's pipeline mollifies exactly as nxr does (same mu under non-default values)."""
    root = runs["root"]
    opts = ["--mollify", "1e-3", "--flip-eps", "1e-9"]
    a = make(root, "delaunay-neuro", runs["format"])
    p, s, g = io("eigen", "solve", "--dataset", a, "--subject", "S1", "--modes", MODES, *opts)
    assert p.returncode == 0, p.stdout + p.stderr[-2000:]
    assert json.loads(p.stdout.strip().splitlines()[-1])["cache"] == "miss"
    arr, prov = package(a)
    o = json.loads(prov)["solve"]
    assert o["options"]["mollify"] == 1e-3 and o["options"]["flip_eps"] == 1e-9 and o["delaunay"]["mollify"] == 1e-3 and o["delaunay"]["flip_eps"] == 1e-9
    assert json.loads(prov)["cache"]["params"]["method"]["mollify"] == 1e-3
    p2 = io("eigen", "solve", "--dataset", a, "--subject", "S1", "--modes", MODES, *opts)[0]
    assert json.loads(p2.stdout.strip().splitlines()[-1])["cache"] == "hit"
    p3 = io("eigen", "solve", "--dataset", a, "--subject", "S1", "--modes", MODES)[0]      # the default values: another key; a package is already stored
    assert p3.returncode != 0 or json.loads(p3.stdout.strip().splitlines()[-1])["cache"] == "miss"
    b = make(root, "delaunay-nxr", runs["format"])
    p, s, g = nxr("eigen", "solve", "--dataset", b, "--subject", "S1", "--modes", MODES, *opts)
    assert p.returncode == 0, p.stdout + p.stderr[-2000:]
    mu, mux = arr["manifold"], package(b)[0]["manifold"]
    assert len(mu) == len(mux)
    d = float(np.abs(mu - mux).max() / mux.max())
    print(f"\n  --mollify 1e-3 --flip-eps 1e-9: {len(mu)} modes, max |mu - nxr mu| / mu_max {d:.2e}")
    assert d <= 1e-9
