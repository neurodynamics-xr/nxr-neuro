"""Writes crossings_golden.txt: the reference of nxr::select::track_crossings::crossings, nxr's `Tiles.crossings` (nxr/detect/vortex_cores.py): the per tile-sense sum of the crossing directions and the crossing counts, on a SYNTHETIC two-hemisphere
cortex (two icospheres of radius 80 mm, 642 vertices each, 100 mm apart) with PLANTED MOVING VORTICES: cores of both senses walk (persistent headings, 4-6 mm a frame) over either sphere, so a tile's steps agree where a walker
crosses it and cancel where walkers of opposite headings do; two walkers share a track id (a track is one walker's steps). The tiles are nxr's surface ladder (level 3: 8 tiles a hemisphere), the anchors nxr's tile_centres. nxr carries each
step's unit direction to its tile's centre (the vector heat method's transport, GeometryKit), makes one unit vector a (tile, sense, track), adds them per tile-sense and embeds the sums at the centres as 3-D tangent vectors. A subset of the
frames is asked for (the stage). Run:
  uv run --project <nxr checkout> python compute/test/golden/gen_crossings_golden.py compute/test/golden/crossings_golden.txt
Written: the mesh, the hemispheres' vertex counts, the tile of each vertex (code 2^level h + t), the normals and the anchors nxr used, the steps (sense, from xyz, displacement xyz, track, frame: nxr's rows), the frames asked for, and nxr's
sums [2T x 3] (tile-sense index 2 code + (sense > 0)) and crossing counts [2T]."""
import sys
from types import SimpleNamespace

import numpy as np

import nxr.detect.vortex_cores as vc
from nxr.measure.tile_connectome import surface_ladder


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(level):
        cache, verts, faces = {}, list(V), []
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = (verts[a] + verts[b]) / 2; verts.append(m / np.linalg.norm(m)); cache[k] = len(verts) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V, F = np.array(verts), np.array(faces)
    return V, F


R, SEP, LEVEL = 80.0, 200.0, 3
U, F0 = icosphere(3)
n0 = len(U)
centres_h = [np.array([-SEP / 2, 0, 0]), np.array([SEP / 2, 0, 0])]
V = np.vstack([R * U + centres_h[0], R * U + centres_h[1]])
F = np.vstack([F0, F0 + n0])
N = np.vstack([U, U])
hemi = np.array(["L"] * n0 + ["R"] * n0)
codes = surface_ladder(V, F, hemi)                                   # [level x V]
lab = codes[LEVEL].astype(np.int64)
vc.LEVELS = (LEVEL,)
cx = SimpleNamespace(support=np.arange(len(V)), Vmm=V, F=F, normals=N, nv=len(V))
tiles = vc.Tiles(cx, {LEVEL: codes[LEVEL]})
rng = np.random.default_rng(5)


def walker(h, s, f0, f1, tid, heading0):
    """a core walking on sphere h from frame f0 to f1 (persistent heading): its steps (sense, from, displacement, track, frame)"""
    p = U[rng.integers(len(U))] * R; hd = heading0
    rows = []
    for f in range(f0, f1 + 1):
        n = p / np.linalg.norm(p)
        t = hd - (hd @ n) * n; t /= np.linalg.norm(t)
        q = p + (4.0 + 2.0 * rng.random()) * t; q = R * q / np.linalg.norm(q)
        if f > f0: rows.append((s, *(p + centres_h[h]), *(q - p), tid, f))
        hd = t + 0.25 * rng.standard_normal(3); p = q
    return rows


steps = []
tid = 1
for k in range(60):
    h = k % 2; s = 1 if (k // 2) % 2 == 0 else -1
    f0 = int(rng.integers(0, 60)); f1 = f0 + int(rng.integers(3, 25))
    # most walkers drift along +z (a consistent direction), some the other way: tiles hold agreeing and cancelling steps
    head = np.array([0.2, 0.1, 1.0 if k % 3 else -1.0])
    steps += walker(h, s, f0, f1, tid, head); tid += 1
S = np.asarray(steps, np.float32).reshape(-1, 9)
S = S[np.argsort(S[:, 8], kind="stable")]
frames = np.arange(5, 70)
sums, n, k, u = tiles.crossings(S, frames, LEVEL)
nt = tiles.nt[LEVEL]
cen = tiles.centre[LEVEL]
with open(sys.argv[1], "w") as out:
    out.write(f"nv {len(V)} nf {len(F)} n0 {n0} level {LEVEL} tiles {nt // 2} steps {len(S)} frames {len(frames)}\n")
    out.write("positions " + " ".join(repr(float(v)) for v in V.ravel()) + "\n")
    out.write("faces " + " ".join(str(int(v)) for v in F.ravel()) + "\n")
    out.write("normals " + " ".join(repr(float(v)) for v in N.ravel()) + "\n")
    out.write("node " + " ".join(str(int(v)) for v in lab) + "\n")
    out.write("anchors " + " ".join(str(int(v)) for v in cen) + "\n")
    out.write("steps " + " ".join(repr(float(v)) for v in S.ravel()) + "\n")
    out.write("frames " + " ".join(str(int(v)) for v in frames) + "\n")
    out.write("sums " + " ".join(repr(float(v)) for v in np.asarray(sums).ravel()) + "\n")
    out.write("n " + " ".join(str(int(v)) for v in n) + "\n")
print(f"wrote {sys.argv[1]}: {len(S)} steps, {int(n.sum())} crossings in {int((n > 0).sum())} of {nt} tile-senses; mean |sum|/n {np.linalg.norm(sums, axis=1)[n > 0].sum() / n.sum():.3f}")
