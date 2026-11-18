"""Goldens for the minimum-norm kernel WITH THE RECORDING'S PROJECTORS (the node meeg.minimum_norm), from synthetic data only.

The case: MNE-Python's sphere forward on 32 MEG channels (16 magnetometers, 16 planar gradiometers, on a 0.12 m sphere), 42 free-orientation sources of an
icosphere at 0.07 m; ONE active SSP projector (a vector over the magnetometers); a full-rank noise covariance C of the data AS RECORDED and the covariance
P C P^T of the PROJECTED data. Both references are handed the gain and the covariance unprojected (or projected) with the projector in their own form,
and apply it themselves:
  MNE-Python   info['projs'] = [the projector]; the covariance's own projs empty (raw) or the projector (projected); mne.cov.regularize (0.1 per type), then
               make_inverse_operator (loose 1, depth 0.8) with its defaults, and the kernel of prepare_inverse_operator / _assemble_kernel (it ends with inv['proj']).
  Brainstorm   gen_kernel_projectors_golden.m: the channel file's Projector (Status 1), process_inverse_2018's own lines (BuildProjector, Proj * Gain,
               Proj * NoiseCov * Proj'), then bst_inverse_linear_2018 (minnorm, amplitude, depth 0.5 / 10, reg 0.1, SNR 3).

Run (nxr's environment), in a work folder:
  uv run --project <nxr checkout> python gen_kernel_projectors_golden.py inputs   <work>
  matlab -batch "gen_kernel_projectors_golden('<work>')"     (one MATLAB run at a time; BRAINSTORM3_HOME set)
  uv run --project <nxr checkout> python gen_kernel_projectors_golden.py assemble <work> kernel_projectors_golden.txt
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

import mne
from mne.io.constants import FIFF
from mne.minimum_norm.inverse import _assemble_kernel, prepare_inverse_operator

sys.path.insert(0, str(Path(__file__).parent))
from gen_kernel_golden import build_info, forward, icosphere, rnd  # noqa: E402

mne.set_log_level("ERROR")
rng = np.random.default_rng(20261118)


def covariance(info, n_samples=4000):
    """the covariance of synthetic data as recorded: correlated channels, per-type scales"""
    M = len(info["ch_names"])
    scale = np.array([1e-13 if t == "mag" else 1e-12 for t in info.get_channel_types()])
    X = rng.standard_normal((M, n_samples)) * (1 + 3 * np.exp(-np.arange(M) / 8.0))[:, None]
    X = (rng.standard_normal((M, M)) * 0.3 + np.eye(M)) @ X
    X = X * scale[:, None]
    X -= X.mean(1, keepdims=True)
    return X @ X.T / (n_samples - 1)


def mne_kernel(info, fwd, C, cov_projs):
    cov = mne.Covariance(C, info["ch_names"], [], cov_projs, 4000)
    cov = mne.cov.regularize(cov, info, mag=0.1, grad=0.1, eeg=0.1, proj=True)
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=1.0, depth=0.8, fixed=False, use_cps=False)
    inv = prepare_inverse_operator(inv, nave=1, lambda2=1.0 / 9, method="MNE")
    K, _, _, _ = _assemble_kernel(inv, None, "MNE", None, use_cps=False)
    return K


def main():
    stage, work = sys.argv[1], sys.argv[2]
    info = build_info(n_mag=16, n_grad=16, n_eeg=0)
    types = info.get_channel_types()
    bst_type = {"mag": "MEG MAG", "grad": "MEG GRAD"}
    M = len(info["ch_names"])
    if stage == "inputs":
        fwd = forward(info, icosphere(1, 0.07))
        fwd["sol"]["data"] = rnd(fwd["sol"]["data"])
        G = fwd["sol"]["data"]
        mags = np.array([t == "mag" for t in types])
        u = np.zeros(M)
        u[mags] = np.sin(1.0 + np.arange(mags.sum()))
        u = rnd(u / np.linalg.norm(u))
        u = u / np.linalg.norm(u)
        proj = mne.Projection(active=True, data=dict(data=u[None, :], col_names=info["ch_names"], row_names=None, nrow=1, ncol=M), desc="ssp",
                              kind=FIFF.FIFFV_PROJ_ITEM_FIELD, explained_var=None)
        with info._unlock():
            info["projs"] = [proj]
        C = rnd(covariance(info))
        P = np.eye(M) - np.outer(u, u)
        Cp = rnd(P @ C @ P.T)
        Cp = (Cp + Cp.T) / 2
        out = dict(G=G, u=u, C_raw=C, C_proj=Cp, types=np.array([bst_type[t] for t in types], dtype=object),
                   K_mne_raw=mne_kernel(info, fwd, C, []), K_mne_proj=mne_kernel(info, fwd, Cp, [proj]))
        sio.savemat(f"{work}/cases.mat", out)
        print("G", G.shape, "K", out["K_mne_raw"].shape)
    else:
        c = sio.loadmat(f"{work}/cases.mat", squeeze_me=True)
        b = sio.loadmat(f"{work}/bst_out.mat", squeeze_me=True)
        code = {"MEG MAG": 0, "MEG GRAD": 1}
        with open(sys.argv[3], "w") as f:
            def emit(name, a):
                a = np.atleast_2d(np.asarray(a, float))
                f.write(f"{name} {a.shape[0]} {a.shape[1]}\n" + " ".join(f"{x:.17g}" for x in a.ravel()) + "\n")
            f.write("# kernel with projectors: synthetic sphere forward (MNE-Python), one active SSP projector; MNE-Python and Brainstorm kernels; see gen_kernel_projectors_golden.py\n")
            emit("G", c["G"])
            emit("u", c["u"][:, None])
            emit("C_raw", c["C_raw"])
            emit("C_proj", c["C_proj"])
            emit("types", np.array([code[t] for t in c["types"]])[:, None])
            for k in ("raw", "proj"):
                emit(f"K_mne_{k}", c[f"K_mne_{k}"])
                emit(f"K_bst_{k}", b[f"K_bst_{k}"])


if __name__ == "__main__":
    main()
