"""golden for test_quadrature: nxr.spectral.lagged (lagged, planes, explained, top_plane, lagged_fraction, im_fraction, im_rows, travel_index, rotary_power,
phase_gradient_flux, standing_surrogate, direction_resultant) on small synthetic inputs. Names here are the nxr names; the C++ ones are in include/nxr/quadrature.h.
Run: cd <nxr checkout> && uv run python <this file> <out.txt>"""
import sys
import numpy as np
from scipy.spatial import ConvexHull
from nxr.spectral.lagged import (lagged, planes, explained, top_plane, lagged_fraction, im_fraction, im_rows, travel_index, rotary_power,
                                 phase_gradient_flux, standing_surrogate, direction_resultant)
from nxr.inverse.fused import FaceOperators

out = open(sys.argv[1], "w")


def put(name, a):
    a = np.atleast_2d(np.asarray(a, float))
    out.write(f"{name} {a.shape[0]} {a.shape[1]}\n" + " ".join(repr(float(v)) for v in a.ravel()) + "\n")


def putc(name, a):
    put(name + "_re", a.real)
    put(name + "_im", a.imag)


rng = np.random.default_rng(11)
N, r = 12, 4
F = rng.standard_normal((N, r)) + 1j * rng.standard_normal((N, r))
putc("F", F)
C = F @ F.conj().T
put("lagged_fraction", lagged_fraction(C))
put("im_fraction", im_fraction(F))
put("im_rows", im_rows(F[:5], F))
Q, S = lagged(F)
put("A_normalised", Q @ S @ Q.T)                      # Im C / ||Im C||_F
put("norm_imC", np.linalg.norm(C.imag))
P = planes(Q, S, 3)
put("sigma", [p[2] for p in P])                       # in units of ||Im C||_F
for i, (x, y, s) in enumerate(P):
    put(f"form{i}", np.outer(x, y) - np.outer(y, x))  # x = p (leads), y = q
put("explained", explained(Q, S, P))
x, y = top_plane(F)
put("top_form", np.outer(x, y) - np.outer(y, x))
u = rng.standard_normal(N) + 1j * rng.standard_normal(N)
putc("u", u)
us = np.exp(0.7j) * rng.standard_normal(N)
putc("u_standing", us)
put("travel_index", travel_index(u))
put("travel_index_standing", travel_index(us))
z = rng.standard_normal(150) + 1j * rng.standard_normal(150)
fs, freqs = 100.0, [8.0, 10.0, 12.5]
put("rotary_fs", fs)
put("rotary_freqs", freqs)
putc("rotary_z", z)
R = rotary_power(z, fs, freqs, 7.0)
put("rotary_pos", R[0])
put("rotary_neg", R[1])

# icosphere level 1 and a window on it
t = (1 + 5 ** .5) / 2
V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
V /= np.linalg.norm(V, axis=1, keepdims=True)
E = set()
for s in ConvexHull(V).simplices:
    for i in range(3):
        E.add(tuple(sorted((s[i], s[(i + 1) % 3]))))
mid = np.array([V[a] + V[b] for a, b in sorted(E)])
V = np.vstack([V, mid / np.linalg.norm(mid, axis=1, keepdims=True)])
Fc = ConvexHull(V).simplices.copy()
for i, f in enumerate(Fc):
    if np.dot(np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]), V[f].mean(0)) < 0:
        Fc[i] = f[[0, 2, 1]]
V = V.astype(np.float32).astype(np.float64)
put("positions", V)
put("faces", Fc.astype(float))
ops = FaceOperators(V, Fc)
Gf = np.stack([np.stack([ops.grad[d][np.arange(len(Fc)), Fc[:, s]].A1 for d in range(3)], 1) for s in range(3)], 1)   # [faces, slot, xyz]
a, b, c = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
Af = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
win = np.where(V[Fc].mean(1)[:, 2] > 0.3)[0]
put("window", win.astype(float))
Z = np.exp(-1j * 2.0 * (V @ np.array([0.6, 0.8, 0.0]))) * (1 + 0.3 * rng.standard_normal(len(V))) + 0.2 * (rng.standard_normal(len(V)) + 1j * rng.standard_normal(len(V)))
putc("Z", Z)
b1, b2 = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
ang, T = phase_gradient_flux(Z, Fc[win], Gf[win], Af[win], b1, b2)
put("flux_angle", ang)
put("flux_strength", T)
Zs = standing_surrogate(Z)
putc("Zs", Zs)
a2, T2 = phase_gradient_flux(Zs, Fc[win], Gf[win], Af[win], b1, b2)
put("flux_strength_surrogate", T2)
angles = rng.vonmises(0.5, 2.0, 40)
put("angles", angles)
Rr, p, md = direction_resultant(angles)
put("dir_R", Rr)
put("dir_p", p)
put("dir_mean", md)
out.close()
