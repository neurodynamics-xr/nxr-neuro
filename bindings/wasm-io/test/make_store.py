"""A synthetic datastore for the wasm-io tests (no real data): two icospheres as a FreeSurfer subject `fsaverage5`, a dataset made by the native CLI
with its default subject (surfaces, frames, group tree, Levi-Civita measurements), and a subject with an OBJ surface.
    python make_store.py <native nxr-neuro> <out folder>      (numpy only)"""
import subprocess, sys
from pathlib import Path
import numpy as np

cli, out = sys.argv[1], Path(sys.argv[2])

def icosphere(levels):
    t = (1 + 5 ** 0.5) / 2
    V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6],
         [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = [list(np.array(v) / np.linalg.norm(v)) for v in V]
    for _ in range(levels):
        mid, F2 = {}, []
        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = (np.array(V[a]) + np.array(V[b])) / 2
                V.append(list(p / np.linalg.norm(p)))
                mid[k] = len(V) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            F2 += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = F2
    return np.array(V), np.array(F, np.int32)

def write_fs(path, V, F):   # FreeSurfer triangle surface (big-endian)
    with open(path, "wb") as f:
        f.write(bytes([0xFF, 0xFF, 0xFE]) + b"created by wasm-io test\n\n")
        f.write(np.array([len(V), len(F)], ">i4").tobytes())
        f.write(np.asarray(V, ">f4").tobytes())
        f.write(np.asarray(F, ">i4").tobytes())

def run(*a):
    p = subprocess.run([cli, *map(str, a)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr

V, F = icosphere(2)                       # 162 vertices a hemisphere
surf = out / "templates" / "fsaverage5" / "surf"
surf.mkdir(parents=True)
rng = np.random.default_rng(1)
for h, cx in (("lh", -40.0), ("rh", 40.0)):
    P = V * (60 + rng.uniform(-3, 3, (len(V), 1))) + [cx, 0, 0]
    for name in ("pial", "white", "inflated"):
        write_fs(surf / f"{h}.{name}", P, F)
    write_fs(surf / f"{h}.sphere.reg", V * 100, F)
store = out / "store"
run("create", "dataset", "--datastore", store, "--name", "d", "--templates", out / "templates", "--atlas-finest", "parts 4", "--lc-scales", "parts 4")
run("create", "subject", "--dataset", store / "d", "--name", "sub-1")
obj = out / "oct.obj"
obj.write_text("v 0.01 0 0\nv -0.01 0 0\nv 0 0.01 0\nv 0 -0.01 0\nv 0 0 0.01\nv 0 0 -0.01\n"
               "f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\nf 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n")
run("add", "surface", "--dataset", store / "d", "--subject", "sub-1", obj)
# a BUMPY sphere of one piece (random radii: no symmetry, so no repeated eigenvalues) — what `eigen solve` is compared on
Vb, Fb = icosphere(2)
Pb = Vb * (0.05 + rng.uniform(-0.004, 0.004, (len(Vb), 1)))
bumpy = out / "bumpy.obj"
bumpy.write_text("".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in Pb.tolist()) + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in Fb.tolist()))
run("add", "surface", "--dataset", store / "d", "--subject", "sub-1", bumpy)
# STATISTIC rows (the bundle's `statistics`: complete rows only, ordered by path then name): the table is added as the writer's ensure_statistics adds it
# (the block of model.sql between its markers), then three rows, one of them pending
import sqlite3, re
sql = (Path(__file__).resolve().parents[3] / "io" / "schema" / "model.sql").read_text()
block = sql[sql.index("-- ── statistic (begin"):sql.index("-- ── statistic (end)")]
con = sqlite3.connect(store / "d" / "dataset.sqlite")
if not con.execute("SELECT 1 FROM sqlite_master WHERE name = 'statistic'").fetchone():
    con.executescript(block)
sid = con.execute("SELECT id FROM subject WHERE name = 'sub-1'").fetchone()[0]
axes = '[{"name":"frequency","size":4,"unit":"Hz"}]'
for i, (name, path, measure, status) in enumerate([("psd_b", "statistic/psd_b", "power_density", "complete"), ("csd_a", "statistic/csd_a", "cross_spectral_density", "complete"),
                                                  ("psd_pending", "statistic/psd_pending", "power_density", "pending")]):
    con.execute("INSERT INTO statistic (id, status, subject_id, name, path, measure, method, value_type, axes_json, n_members) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"stat-{i}", status, sid, name, path, measure, "welch", "scalar", axes, 3))
con.commit(); con.close()
print(store)
