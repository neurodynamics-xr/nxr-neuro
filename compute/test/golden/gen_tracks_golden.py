"""Writes tracks_golden.txt: the reference of nxr::select::Linker, the linking of nxr's `_track_stats` (nxr/detect/vortex.py), on PLANTED MOVING VORTICES: on a sphere of radius 80 mm, a few cores of both senses
walk (random headings, about 4-6 mm a frame) for many frames, appear and disappear, two same-sense cores pass within the link distance of each other (the greedy choice decides who follows whom), a few lone cores
come and go, and the run has frames with no core and frames left out (a gap breaks every track). `_track_stats` is called as it is (its Jaccard sums empty) with `steps=[]`: the steps it appends are the links
(sense, from, displacement, track, frame), and its lifetime, path and displacement lists are of the tracks of at least 3 frames. Run:
  uv run --project <nxr checkout> python compute/test/golden/gen_tracks_golden.py compute/test/golden/tracks_golden.txt
Written: the cores (frame, sense, centre in mm, the order nxr is given: by frame, sense +1 first), the link distance, nxr's steps, and its (frames, path, displacement) of each track of >= 3 frames, sorted."""
import sys

import numpy as np
from nxr.detect.vortex import CORE_LINK, _track_stats

rng = np.random.default_rng(23)
R = 80.0
nfr = 90
gone = {30, 31, 62}                                     # frames left out of the run (bad): a gap
empty = {45, 46}                                        # good frames with no core at all


def on_sphere(p):
    return R * p / np.linalg.norm(p)


walkers = []                                            # (sense, start frame, end frame, position, heading)
for k in range(9):
    s = 1 if k % 2 == 0 else -1
    a = int(rng.integers(0, 50)); b = a + int(rng.integers(6, 40))
    p = on_sphere(rng.standard_normal(3))
    walkers.append([s, a, b, p, rng.standard_normal(3)])
# two same-sense cores passing close: the first walks past the second, 10 mm apart, both in frames 10-24
q = on_sphere(np.array([1.0, 0.2, 0.3])); walkers.append([1, 10, 24, q, np.array([0.0, 1.0, 0.0])])
walkers.append([1, 10, 24, on_sphere(q + np.array([0.0, 0.0, 10.0])), np.array([0.0, 1.0, 0.0])])
# lone cores: a frame each
lone = [(int(rng.integers(0, nfr)), int(rng.choice([-1, 1])), on_sphere(rng.standard_normal(3))) for _ in range(12)]
cores = []                                              # (frame, sense, order, centre)
pos = [w[3].copy() for w in walkers]
for f in range(nfr):
    if f in gone or f in empty:
        for i, w in enumerate(walkers):                  # the walkers move on regardless
            h = w[4] - (w[4] @ pos[i]) / (pos[i] @ pos[i]) * pos[i]; pos[i] = on_sphere(pos[i] + 5.0 * h / np.linalg.norm(h)); w[4] = h + 0.3 * rng.standard_normal(3)
        continue
    for i, w in enumerate(walkers):
        h = w[4] - (w[4] @ pos[i]) / (pos[i] @ pos[i]) * pos[i]
        pos[i] = on_sphere(pos[i] + (4.0 + 2.0 * rng.random()) * h / np.linalg.norm(h)); w[4] = h + 0.3 * rng.standard_normal(3)
        if w[1] <= f <= w[2]:
            cores.append((f, w[0], len(cores), pos[i].copy() + 0.4 * rng.standard_normal(3)))
    for lf, ls, lp in lone:
        if lf == f:
            cores.append((f, ls, len(cores), lp))
cores.sort(key=lambda c: (c[0], -c[1], c[2]))           # by frame, sense +1 first, then as made
fr_all = np.array([f for f in range(nfr) if f not in gone], np.int64)
cf = np.array([c[0] for c in cores], np.int64); cs = np.array([c[1] for c in cores]); cc = np.array([c[3] for c in cores]); ca = np.ones(len(cores))
pos_in_fr = np.searchsorted(fr_all, cf)                # nxr's patches name their frame by its POSITION among the run's good frames (`fr`), not its number
parts = [(pos_in_fr, cs, cc, ca)]
H, P = 9, 18
steps = []
empty_j = {"half": (np.zeros(0), np.zeros(0)), "cycle": (np.zeros(0), np.zeros(0))}
out = _track_stats(0.0, [np.zeros(1), np.zeros(1)], empty_j, parts, fr_all, 10.0, ca, 250.0, H, P, steps)
# the tracks of >= 3 frames from the steps: a track's frames = its steps + 1; path = sum of the steps' lengths; displacement from the first from to the last from + disp
by = {}
for sg, fx, fy, fz, dx, dy, dz, tid, k in steps:
    t = by.setdefault(int(tid), dict(n=1, path=0.0, c0=np.array([fx, fy, fz]), c=np.array([fx, fy, fz])))
    t["n"] += 1; t["path"] += float(np.linalg.norm([dx, dy, dz])); t["c"] = np.array([fx + dx, fy + dy, fz + dz])
tr = sorted((t["n"], t["path"], float(np.linalg.norm(t["c"] - t["c0"]))) for t in by.values() if t["n"] >= 3)
# cross-check against what nxr reports itself (its median lifetime and path statistics are over the same tracks)
assert len(tr) == out["n_tracks"], (len(tr), out["n_tracks"])
assert abs(np.median([t[0] for t in tr]) / 250.0 * 1000 - out["life_ms"]) < 1e-9
with open(sys.argv[1], "w") as f:
    f.write(f"cores {len(cores)} steps {len(steps)} tracks3 {len(tr)} link {CORE_LINK!r}\n")
    f.write("frame " + " ".join(str(int(v)) for v in cf) + "\n")
    f.write("sense " + " ".join(str(int(v)) for v in cs) + "\n")
    f.write("centre " + " ".join(repr(float(v)) for v in cc.ravel()) + "\n")
    f.write("steps " + " ".join(repr(float(v)) for s in steps for v in s) + "\n")     # 9 numbers a step: sense, from xyz, displacement xyz, track (1-based, by creation), frame
    f.write("tracks3 " + " ".join(repr(float(v)) for t in tr for v in t) + "\n")      # n_frames, path, displacement of each track of >= 3 frames, sorted
print(f"wrote {sys.argv[1]}: {len(cores)} cores, {len(steps)} links, {len(tr)} tracks of >= 3 frames, senses {np.bincount((cs > 0).astype(int))}")
