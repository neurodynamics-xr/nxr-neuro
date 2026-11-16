"""Goldens for the unconstrained minimum-norm kernel (nxr::meeg), from synthetic data only.

Stage "inputs":   builds the cases (a gain from MNE-Python's sphere forward on synthetic sensors and sources, a noise covariance, a projector) and writes
                  <work>/cases.mat for MATLAB, with MNE-Python's kernel for each case.
Stage "assemble": reads <work>/bst_out.mat (written by gen_kernel_golden.m: Brainstorm's output for the same cases) and writes kernel_golden.txt.

Run (nxr's environment):
  uv run --project <nxr checkout> python gen_kernel_golden.py inputs   <work>
  matlab -batch "gen_kernel_golden('<work>')"           (one MATLAB run at a time; Brainstorm on the path)
  uv run --project <nxr checkout> python gen_kernel_golden.py assemble <work> kernel_golden.txt
The gain is a real forward model (MNE-Python's sphere model: 12 magnetometers and 12 planar gradiometers on a 0.12 m sphere, 10 EEG electrodes at 0.09 m,
42 sources of an icosphere at 0.07 m, free orientation), not a random matrix.
"""
import sys

import numpy as np
import scipy.io as sio

import mne
from mne.io.constants import FIFF
from mne.minimum_norm.inverse import _assemble_kernel, prepare_inverse_operator

mne.set_log_level("ERROR")
rng = np.random.default_rng(20261110)


def icosphere(levels, radius):
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
    return radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def rnd(a):
    a = np.asarray(a, float)
    return np.array([float(f"{x:.12g}") for x in a.ravel()]).reshape(a.shape)


def fibonacci(n, radius):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return radius * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def build_info(n_mag=12, n_grad=12, n_eeg=10):
    names = [f"MAG{i:02d}" for i in range(n_mag)] + [f"GRAD{i:02d}" for i in range(n_grad)] + [f"EEG{i:02d}" for i in range(n_eeg)]
    types = ["mag"] * n_mag + ["grad"] * n_grad + ["eeg"] * n_eeg
    info = mne.create_info(names, 1000.0, types)
    with info._unlock():
        info["dev_head_t"] = mne.transforms.Transform("meg", "head")
    pos = np.vstack([fibonacci(n_mag, 0.12), fibonacci(n_grad, 0.121), fibonacci(n_eeg, 0.09)])
    for k, ch in enumerate(info["chs"]):
        p = pos[k]
        z = p / np.linalg.norm(p)
        x = np.cross(z, [0.3, 0.5, 0.8]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        ch["loc"][:3] = p
        ch["loc"][3:12] = np.concatenate([x, y, z])
        if types[k] == "mag":
            ch["coil_type"] = FIFF.FIFFV_COIL_VV_MAG_T3
        elif types[k] == "grad":
            ch["coil_type"] = FIFF.FIFFV_COIL_VV_PLANAR_T1
        else:
            ch["coil_type"] = FIFF.FIFFV_COIL_EEG
            ch["loc"][3:6] = 0  # EEG: reference electrode location unset
    return info


def forward(info, src_pos):
    nn = src_pos / np.linalg.norm(src_pos, axis=1, keepdims=True)
    src = mne.setup_volume_source_space(pos=dict(rr=src_pos, nn=nn), sphere=(0, 0, 0, 0.09), verbose=False)
    sph = mne.make_sphere_model(r0=(0, 0, 0), head_radius=0.09)
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sph, meg=True, eeg=True, mindist=0)
    return fwd


def covariance(info, n_samples=4000):
    """A full-rank covariance with correlated channels, per-type scales."""
    M = len(info["ch_names"])
    scale = np.array([1e-13 if t == "mag" else 1e-12 if t == "grad" else 1e-6 for t in info.get_channel_types()])
    A = rng.standard_normal((M, M)) * (1 + 3 * np.exp(-np.arange(M) / 8.0))[None, :]
    C = A @ A.T / M
    d = 1 / np.sqrt(np.diag(C))
    C = C * d[:, None] * d[None, :]
    return C * scale[:, None] * scale[None, :]


def mne_kernel(info, fwd, C, projs=None, depth=0.8, **kw):
    """MNE-Python's kernel [3N x M], for the data as recorded (projectors inside), the regularisation of mne.cov.regularize (ridge 0.1 per type)."""
    info = info.copy()
    if projs:
        with info._unlock():
            info["projs"] = projs
    cov = mne.Covariance(C, info["ch_names"], [], [], 4000)
    cov = mne.cov.regularize(cov, info, mag=0.1, grad=0.1, eeg=0.1, proj=bool(projs), rank="info" if projs else None)
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=1.0, depth=depth, fixed=False, rank="info" if projs else None, use_cps=False)
    inv = prepare_inverse_operator(inv, nave=1, lambda2=1.0 / 9, method="MNE")
    K, _, _, _ = _assemble_kernel(inv, None, "MNE", None, use_cps=False)
    return K, inv["nsource"]


def ssp_projs(info, n=2):
    mags = [i for i, t in enumerate(info.get_channel_types()) if t == "mag"]
    M = len(info["ch_names"])
    out = []
    for k in range(n):
        v = np.zeros(M)
        v[mags] = np.sin(1.0 + k + np.arange(len(mags))) + 0.3 * k
        out.append(v)
    q, _ = np.linalg.qr(np.stack(out, 1))
    projs = []
    for k in range(n):
        v = q[:, k]
        d = dict(data=v[None, :], col_names=info["ch_names"], row_names=None, nrow=1, ncol=len(v))
        projs.append(mne.Projection(active=True, data=d, desc=f"ssp{k}", kind=FIFF.FIFFV_PROJ_ITEM_FIELD, explained_var=None))
    return projs, q


def main():
    stage, work = sys.argv[1], sys.argv[2]
    info = build_info()
    types_mne = info.get_channel_types()
    bst_type = {"mag": "MEG MAG", "grad": "MEG GRAD", "eeg": "EEG"}
    if stage == "inputs":
        src_pos = icosphere(1, 0.07)
        fwd = forward(info, src_pos)
        fwd["sol"]["data"] = rnd(fwd["sol"]["data"])
        G = fwd["sol"]["data"]  # [M x 3N], x y z per source, rounded to 12 digits: both references read the same numbers as the golden file
        assert G.shape == (len(info["ch_names"]), 3 * len(src_pos)), G.shape
        C = rnd(covariance(info))
        cases = {}
        projs, q = ssp_projs(info)
        M = len(info["ch_names"])
        eeg = np.array([t == "eeg" for t in types_mne])
        A = np.eye(M)
        A[np.ix_(eeg, eeg)] -= 1.0 / eeg.sum()
        P = (np.eye(M) - q @ q.T) @ A
        meg = ~eeg
        for name, sel, deficient in (("full_megeeg", np.ones(M, bool), False), ("full_meg", meg, False), ("deficient_megeeg", np.ones(M, bool), True),
                                     ("deficient_meg", meg, True)):
            idx = np.where(sel)[0]
            sub_info = mne.pick_info(info, idx, copy=True)
            sub_fwd = mne.pick_channels_forward(fwd, [info["ch_names"][i] for i in idx], ordered=True)
            # the projector, as a [M x M] on the picked channels
            pj = None
            if deficient:
                Pm = P[np.ix_(idx, idx)]
                if not sel.all():
                    # the average reference acts on EEG only; with MEG only, just the SSP vectors
                    Pm = np.eye(len(idx)) - q[idx][:, :] @ q[idx].T
                Gp = Pm @ G[idx]
                Cp = Pm @ C[np.ix_(idx, idx)] @ Pm.T
                Gp, Cp = rnd(Gp), rnd((Cp + Cp.T) / 2)
                pj = ssp_projs(sub_info)[0]
                if sel.all():
                    pj = pj + [mne.Projection(active=True, data=dict(data=(np.eye(len(idx))[eeg[idx]].sum(0) / eeg.sum())[None, :], col_names=sub_info["ch_names"], row_names=None, nrow=1, ncol=len(idx)),
                                              desc="average", kind=FIFF.FIFFV_PROJ_ITEM_EEG_AVREF, explained_var=None)]
                K, ns = mne_kernel(sub_info, sub_fwd, Cp, projs=pj)
            else:
                Gp, Cp = G[idx], C[np.ix_(idx, idx)]
                K, ns = mne_kernel(sub_info, sub_fwd, Cp)
            Kn = mne_kernel(sub_info, sub_fwd, Cp, depth=None, projs=pj if deficient else None)[0]
            cases[name] = dict(K_mne_nodepth=Kn if Kn is not None else np.zeros((0, 0)), G=Gp, C=Cp, types=np.array([bst_type[types_mne[i]] for i in idx], dtype=object), K_mne=K)
            print(name, Gp.shape, K.shape)
        sio.savemat(f"{work}/cases.mat", cases)
    else:
        cases = sio.loadmat(f"{work}/cases.mat", squeeze_me=True, struct_as_record=False)
        bst = sio.loadmat(f"{work}/bst_out.mat", squeeze_me=True, struct_as_record=False)
        code = {"MEG MAG": 0, "MEG GRAD": 1, "EEG": 2}
        with open(sys.argv[3], "w") as f:
            def emit(name, a):
                a = np.atleast_2d(np.asarray(a, float))
                f.write(f"{name} {a.shape[0]} {a.shape[1]}\n" + " ".join(f"{x:.12g}" for x in a.ravel()) + "\n")
            f.write("# kernel golden: synthetic sphere forward (MNE-Python), MNE-Python kernels, Brainstorm kernels; see gen_kernel_golden.py\n")
            for name, c in cases.items():
                if name.startswith("__") or not hasattr(c, "G"):
                    continue
                emit(f"{name}.G", c.G)
                emit(f"{name}.C", c.C)
                emit(f"{name}.types", np.array([code[t] for t in np.atleast_1d(c.types)])[:, None])
                emit(f"{name}.K_mne", c.K_mne)
                if np.size(c.K_mne_nodepth):
                    emit(f"{name}.K_mne_nodepth", c.K_mne_nodepth)
                for key in bst:
                    if key.startswith(name + "__"):
                        v = key.split("__")[1]
                        emit(f"{name}.bst_{v}.K", bst[key].K)
                        emit(f"{name}.bst_{v}.W", bst[key].W)


if __name__ == "__main__":
    main()
