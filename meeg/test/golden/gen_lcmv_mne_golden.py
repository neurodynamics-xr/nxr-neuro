"""Goldens for the LCMV beamformer under MNE-Python's conventions (nxr::meeg::rankWhitener, nxr::meeg::mneLcmvKernel; `run lcmv --conventions mne`), from synthetic data only.

The reference is MNE-Python itself: mne.beamformer.make_lcmv, its defaults (reg 0.05, rank 'info', depth None, inversion 'matrix', reduce_rank False) and the options pick_ori
(None, 'normal', 'max-power') x weight_norm (None, 'unit-noise-gain', 'unit-noise-gain-invariant', 'nai'), with and without a noise covariance. The stored operator is the matrix
apply_lcmv applies to the channels as recorded: filters['weights'] @ filters['whitener'] (the whitener holds the projectors).

The case: an EEG cap of 30 electrodes on MNE-Python's four-shell sphere model (head radius 0.09 m), 200 discrete sources inside 0.06 m with random unit orientations, the
average-reference projector in info['projs'] (not applied: MNE-Python uses every projector of the info); a data covariance of synthetic recordings (the sources' activity through
the gain, plus correlated channel noise) and a full-rank noise covariance of correlated channel noise. Free orientation uses the forward in head coordinates (surf_ori False: the rows
are x, y, z); normal and max-power use the same forward in surface orientation (surf_ori True: MNE-Python requires it for 'normal', and it aligns the max-power orientation's sign
with the source normal), whose third column of a source is the gain along its normal.

Written: the inputs (G [channels x 3 sources], the normals, both covariances, the projector, the rank) and, per case, the kernel rows of 16 sampled sources (3 rows a source for free
orientation) and, for max-power, the orientations (in head coordinates) of the sampled sources; the whitener once with the noise covariance and
once without (the ad hoc identity covariance MNE-Python makes when none is given: the projector).

Run, in a throwaway environment (never a project's):
  uv venv <venv> && VIRTUAL_ENV=<venv> uv pip install mne==<version>
  <venv>/bin/python gen_lcmv_mne_golden.py lcmv_mne_golden.txt
"""
import sys

import numpy as np

import mne
from mne.beamformer import make_lcmv
from mne._fiff.proj import make_eeg_average_ref_proj, make_projector
from mne.transforms import Transform

mne.set_log_level("ERROR")
rng = np.random.default_rng(20261120)

M, N, T = 30, 200, 3000
SAMPLED = np.sort(rng.choice(N, 16, replace=False))


def cap():
    """30 electrodes on the upper part of a 0.09 m sphere"""
    th = np.arccos(rng.uniform(-0.2, 1.0, M))
    ph = rng.uniform(0, 2 * np.pi, M)
    pos = 0.09 * np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    names = [f"E{i:02d}" for i in range(M)]
    info = mne.create_info(names, 250.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, pos)), coord_frame="head"))
    return info


def sources():
    """200 points inside 0.06 m, random unit orientations"""
    r = 0.06 * rng.uniform(0.2, 1.0, N) ** (1 / 3)
    d = rng.standard_normal((N, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nn = rng.standard_normal((N, 3))
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    return mne.setup_volume_source_space(pos=dict(rr=r[:, None] * d, nn=nn))


def main(out):
    info = cap()
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.09)
    src = sources()
    fwd = mne.make_forward_solution(info, trans=Transform("head", "mri"), src=src, bem=sphere, meg=False, eeg=True)
    fwd_surf = mne.convert_forward_solution(fwd, surf_ori=True)
    G = fwd["sol"]["data"]
    nn = src[0]["nn"][src[0]["vertno"]]
    proj = make_eeg_average_ref_proj(info, activate=False)
    with info._unlock():
        info["projs"] = [proj]
    P, _, _ = make_projector(info["projs"], info["ch_names"])
    names = info["ch_names"]
    # the recordings: every source along its orientation with its own activity (three strong ones), and correlated channel noise; the noise recording: channel noise only
    S = 1e-9 * rng.standard_normal((N, T))
    for k, f in zip((11, 77, 150), (10.0, 10.5, 23.0)):
        S[k] += 2e-8 * np.sin(2 * np.pi * f * np.arange(T) / 250.0 + k)
    Gn = np.einsum("mkc,kc->mk", G.reshape(M, N, 3), nn)
    mix = np.eye(M) + 0.3 * rng.standard_normal((M, M))
    X = Gn @ S + 1e-6 * mix @ rng.standard_normal((M, T))
    E = 1e-6 * (np.eye(M) + 0.2 * rng.standard_normal((M, M))) @ rng.standard_normal((M, T))
    Cd = np.cov(X)
    Cn = np.cov(E)
    data_cov = mne.Covariance(Cd, names, [], [], T - 1)
    noise_cov = mne.Covariance(Cn, names, [], [], T - 1)

    lines = [f"# lcmv mne golden: MNE-Python {mne.__version__} (numpy {np.__version__}), mne.beamformer.make_lcmv; see gen_lcmv_mne_golden.py"]

    def put(name, a):
        a = np.atleast_2d(np.real(np.asarray(a)).astype(float))
        lines.append(f"{name} {a.shape[0]} {a.shape[1]}")
        lines.append(" ".join(f"{v:.15g}" for v in a.ravel()))

    put("G", G)
    put("nn", nn)
    put("Cd", Cd)
    put("Cn", Cn)
    put("P", P)
    put("sampled", SAMPLED[None, :])
    rank = None
    cases = []
    whiteners = set()
    for noise in (True, False):
        for ori in (None, "normal", "max-power"):
            for wn in (None, "unit-noise-gain", "unit-noise-gain-invariant", "nai"):
                if not noise and (ori, wn) not in ((None, "unit-noise-gain-invariant"), (None, None), ("normal", "unit-noise-gain"), ("max-power", "nai")):
                    continue
                f = make_lcmv(info, fwd if ori is None else fwd_surf, data_cov, reg=0.05, noise_cov=noise_cov if noise else None, pick_ori=ori, weight_norm=wn, rank="info")
                K = f["weights"] @ f["whitener"]
                n_orient = 3 if ori is None else 1
                rows = (SAMPLED[:, None] * n_orient + np.arange(n_orient)[None, :]).ravel()
                key = f"{'noise' if noise else 'adhoc'}.{ori or 'free'}.{wn or 'none'}"
                cases.append(key)
                put(key + ".K", K[rows])
                if f"{key.split('.')[0]}.W" not in whiteners:   # the whitener depends on the noise covariance only
                    whiteners.add(f"{key.split('.')[0]}.W")
                    put(f"{key.split('.')[0]}.W", f["whitener"])
                if ori == "max-power":
                    # the orientation in the surface frame (x, y tangential, z the normal) back to head coordinates: rows of the source's frame
                    frames = fwd_surf["source_nn"].reshape(N, 3, 3)
                    put(key + ".ori", np.einsum("kc,kcd->kd", f["max_power_ori"], frames)[SAMPLED])
                rank = f["rank"]
    lines.insert(1, f"# cases: {' '.join(cases)}")
    put("rank", [[rank]])
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
