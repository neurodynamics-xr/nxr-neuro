"""Goldens for the sphere forward models (nxr::meeg forward.h, sphere.h), from synthetic data only.

Stage "inputs":   builds the case (two head surfaces, 13 MEG channels with several integration points each, 8 EEG electrodes, 43 sources, a few sources outside the
                  spheres), computes MNE-Python's references, and writes <work>/cases.mat for MATLAB.
Stage "assemble": reads <work>/bst_out.mat (written by gen_forward_golden.m: Brainstorm's output for the same case) and writes forward_golden.txt.

Run (nxr's environment):
  uv run --project <nxr checkout> python gen_forward_golden.py inputs   <work>
  matlab -batch "gen_forward_golden('<work>')"          (one MATLAB run at a time; Brainstorm on the path)
  uv run --project <nxr checkout> python gen_forward_golden.py assemble <work> forward_golden.txt
Inputs are rounded to 12 significant digits first, so that both references and nxr read the same numbers as the golden file.
"""
import sys

import numpy as np
import scipy.io as sio

import mne
from mne.bem import _fwd_eeg_get_multi_sphere_model_coeffs as multi_sphere_coeffs
from mne.forward._compute_forward import _eeg_spherepot_coil, _sphere_field

mne.set_log_level("ERROR")
rng = np.random.default_rng(5)


def icos(levels):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6),
         (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(x, float) for x in v]
    for _ in range(levels):
        mid, g = {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                mid[k] = len(v)
                v.append((v[a] + v[b]) / 2)
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            g += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = g
    v = np.array(v)
    return v / np.linalg.norm(v, axis=1, keepdims=True), np.array(f)


def rnd(a):
    a = np.asarray(a, float)
    return np.array([float(f"{x:.12g}") for x in a.ravel()]).reshape(a.shape)


def frame(p):
    z = p / np.linalg.norm(p)
    x = np.cross(z, [0.3, 0.5, 0.8])
    x /= np.linalg.norm(x)
    return x, np.cross(z, x), z


def fib(n, r, off):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 1.6 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return off + r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def build():
    c0 = np.array([0.003, -0.002, 0.045])
    u, F = icos(2)
    scalp = c0 + u * np.array([0.092, 0.084, 0.088]) * (1 + 0.03 * np.sin(3 * u[:, [0]] + u[:, [2]]))
    inner = c0 + np.array([0, 0, -.002]) + u * np.array([0.076, 0.069, 0.071]) * (1 + 0.04 * np.cos(2 * u[:, [1]] + 3 * u[:, [2]]))
    chs = []

    def mk(loc, ori, w, typ, name):
        chs.append(dict(Loc=np.array(loc).T, Orient=np.array(ori).T, Weight=np.array(w, float), Type=typ, Name=name))
    for k, p in enumerate(fib(5, 0.11, c0)):                     # magnetometers: four integration points
        x, y, z = frame(p - c0)
        mk([p + 0.01 * (sx * x + sy * y) for sx in (-1, 1) for sy in (-1, 1)], [z] * 4, [0.25] * 4, "MEG MAG", f"M{k}")
    for k, p in enumerate(fib(4, 0.105, c0)):                    # axial gradiometers: two coils of four points, 5 cm apart
        x, y, z = frame(p - c0)
        b = 0.05
        pts = [p + 0.01 * (sx * x + sy * y) for sx in (-1, 1) for sy in (-1, 1)]
        mk(pts + [q + b * z for q in pts], [z] * 8, [1 / (4 * b)] * 4 + [-1 / (4 * b)] * 4, "MEG GRAD", f"G{k}")
    for k, p in enumerate(fib(4, 0.108, c0)):                    # planar gradiometers: two points
        x, y, z = frame(p - c0)
        mk([p + 0.008 * x, p - 0.008 * x], [z, z], [1 / 0.016, -1 / 0.016], "MEG GRAD", f"P{k}")
    ec = c0 + [0.001, 0, 0.002]
    eeg = fib(8, 0.09, ec)
    src, _ = icos(1)
    src = c0 + src * np.array([0.062, 0.056, 0.058]) * (0.8 + 0.5 * rng.random((len(src), 1)))
    src = np.vstack([src, c0 + [0.001, 0.002, -0.001]])
    out = ec + np.array([[0.08, 0, 0], [0, 0.085, 0], [0, 0, 0.095]])      # beyond the innermost shell, the outer shell and the electrodes' sphere
    return dict(c0=c0, ec=ec, scalp=scalp, inner=inner, F=F, chs=chs, eeg=eeg, src=rnd(src), src_out=np.vstack([rnd(src[:2]), out]))


def flat(chs):
    pos = np.concatenate([c["Loc"].T.reshape(-1, 3) for c in chs])
    ori = np.concatenate([c["Orient"].T.reshape(-1, 3) for c in chs])
    w = np.concatenate([np.atleast_1d(c["Weight"]) for c in chs])
    ch = np.concatenate([[i] * len(np.atleast_1d(c["Weight"])) for i, c in enumerate(chs)])
    return pos, ori, w, ch


def main():
    stage, work = sys.argv[1], sys.argv[2]
    d = build()
    chs = d["chs"]
    for c in chs:
        c["Loc"], c["Orient"] = rnd(c["Loc"]), rnd(c["Orient"])
    for k in ("c0", "ec", "scalp", "inner", "eeg"):
        d[k] = rnd(d[k])
    pos, ori, w, ch = flat(chs)
    eeg_c = (d["eeg"], None, np.ones(len(d["eeg"])), np.arange(len(d["eeg"])))
    if stage == "inputs":
        dt = [("Loc", "O"), ("Orient", "O"), ("Weight", "O"), ("Type", "O"), ("Name", "O")]
        arr = np.empty(len(chs), dtype=dt)
        for i, c in enumerate(chs):
            arr[i] = (c["Loc"], c["Orient"], c["Weight"], c["Type"], c["Name"])
        sio.savemat(f"{work}/cases.mat", dict(chs=arr, src=d["src"], src_out=d["src_out"], c0=d["c0"], ec=d["ec"], eeg=d["eeg"], inner=d["inner"], scalp=d["scalp"], F=d["F"] + 1))
        return
    bst = sio.loadmat(f"{work}/bst_out.mat", squeeze_me=True, struct_as_record=False)
    mne_meg = _sphere_field(d["src"], (pos, ori, w, ch), dict(r0=d["c0"]))   # [3N, M]
    mne_meg_out = _sphere_field(d["src_out"], (pos, ori, w, ch), dict(r0=d["c0"]))
    sets = {}
    for name, rr, sg in (("bst", [0.88, 0.93, 1.0], [0.33, 0.0042, 0.33]), ("mne", [0.9, 0.92, 0.97, 1.0], [0.33, 1.0, 0.004, 0.33])):
        m = mne.make_sphere_model(r0=tuple(d["ec"]), head_radius=0.09, relative_radii=rr, sigmas=sg)
        G = _eeg_spherepot_coil(d["src"], eeg_c, m).T
        sets[name] = dict(mu=m["mu"], lam=m["lambda"] * sg[-1], G=G, f=multi_sphere_coeffs(dict(layers=[dict(rel_rad=r, sigma=s) for r, s in zip(rr, sg)]), 201))
    with open(sys.argv[3], "w") as f:
        def emit(name, a):
            a = np.atleast_2d(np.asarray(a, float))
            f.write(f"{name} {a.shape[0]} {a.shape[1]}\n" + " ".join(f"{x:.12g}" for x in a.ravel()) + "\n")
        f.write("# forward golden: synthetic case, Brainstorm (bst_meg_sph, bst_os, bst_eeg_sph, bst_berg) and MNE-Python (_sphere_field, _eeg_spherepot_coil); see gen_forward_golden.py\n")
        for k, v in (("src", d["src"]), ("src_out", d["src_out"]), ("c0", d["c0"][None, :]), ("ec", d["ec"][None, :]), ("eeg", d["eeg"]), ("inner", d["inner"]), ("scalp", d["scalp"]),
                     ("coil_pos", pos), ("coil_ori", ori), ("coil_w", w[:, None]), ("coil_ch", ch[:, None])):
            emit(k, v)
        emit("meg_single.mne", mne_meg.T)
        emit("meg_single_out.mne", mne_meg_out.T)
        for k in ("meg_single", "meg_single_out", "os_centre", "os_radius", "meg_os", "os_scalp_centre", "os_scalp_radius", "eeg_bst", "eeg_bst_out", "eeg_bst_1", "berg_mu", "berg_lam", "berg_f"):
            emit(f"{k}.bst", np.asarray(bst[k], float).reshape(-1, 1) if np.ndim(bst[k]) <= 1 else bst[k])
        for name, s in sets.items():
            emit(f"eeg_{name}layers.mne_mu", s["mu"][:, None])
            emit(f"eeg_{name}layers.mne_lam", s["lam"][:, None])
            emit(f"eeg_{name}layers.mne", s["G"])
            emit(f"eeg_{name}layers.f", s["f"][:, None])


if __name__ == "__main__":
    main()
