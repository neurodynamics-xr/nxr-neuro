# The golden of nxr::stats::networkSignFlip, numpyChoice and distanceMatchedControl (compute/test/test_network_flip.cpp): the reference's flip and control
# (nxr's flow through the fibre network, flownet.py) on generated values, with one numpy default_rng(0) shared in the reference's order: per frequency the flip, then the
# control's draws per occupied distance bin. Run: python gen_network_flip_golden.py > network_flip_golden.txt
import numpy as np
rng = np.random.default_rng(0); perms = 200; S, E = 12, 40
e = np.arange(E); joined = (e * 7 % 5) < 2; dist = (e * 37 % 300).astype(float)
bins = np.digitize(dist, [0, 30, 45, 60, 80, 100, 130, 170, 250])
def flip(Fs):                                   # the reference's, verbatim
    obs = Fs.sum(0); E_ = np.where(rng.random((perms, len(Fs))) < .5, -1., 1.); nl = E_ @ Fs
    p = (np.sum(np.abs(nl) >= np.abs(obs), 0) + 1) / (perms + 1); mx = np.abs(nl / np.maximum(np.abs(Fs).sum(0), 1e-30)).max(1)
    pf = (np.sum(mx[:, None] >= np.abs(obs / np.maximum(np.abs(Fs).sum(0), 1e-30))[None], 0) + 1) / (perms + 1); return obs, p, pf
print("bins", *bins)
for k in range(3):
    s = np.arange(S)[:, None]
    Fs = np.sin(1.7 * s + 0.3 * e[None] ** 2 + k) + 0.2 * np.cos(e[None] + k) + 0.15
    obs, p, pf = flip(Fs)
    ctrl = np.zeros(E, bool)
    for b in np.unique(bins[joined]):
        cand = np.flatnonzero(~joined & (bins == b)); need = np.sum(joined & (bins == b))
        if len(cand): ctrl[rng.choice(cand, min(len(cand), need), replace=False)] = True
    print("p", k, *[f"{v:.17g}" for v in p]); print("pf", k, *[f"{v:.17g}" for v in pf]); print("ctrl", k, *ctrl.astype(int))
r2 = np.random.default_rng(5); r2.random(); r2.integers(0, 2 ** 32, dtype=np.uint32)   # leave a 32-bit half buffered
print("choice", *r2.choice(500, 37, replace=False)); print("choice", *r2.choice(9, 9, replace=False)); print("choice", *r2.choice(3, 0, replace=False)); print("after", f"{r2.random():.17g}")
