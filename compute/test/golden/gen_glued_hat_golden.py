"""Writes glued_hat_golden.txt: the reference of apply_kernel's `form` ON THE GLUED SURFACE, nxr's `vortex.Cortex`:
  the curl on the native mesh's faces (FaceOperators.div_curl(tangent(q)), zero on the medial wall), placed on the glued mesh's kept vertices (zero on the bridge), then the Mexican hat
  g(lam) = lam s2 exp(-lam s2 / 2) of `vortex_cores.hat_operator` applied through the glued mesh's Laplace-Beltrami modes (`Cortex.coords`, the gain, `P.T`), read at EVERY glued vertex.
On a SYNTHETIC glued pair of spheres (two icospheres of radius 0.08 m, 0.2 m apart, a medial-wall cap on each cut and bridged by nxr's `glue_mesh`: the Studio's glue) with planted currents (a rigid
rotation about the hemisphere's own axis plus a normal part, and two smooth fields). The modes are the dense generalised eigenproblem of nxr's intrinsic-Delaunay cotan Laplacian and mixed-Voronoi mass of
the glued mesh (the package's pencil). The gain is scaled by its ANALYTIC peak (2/e), as nxr::signal::graphfilterbank::mexicanHat: nxr scales by the sample maximum of its dense spectrum. Levels 2 and 3. Run:
  uv run --project <nxr checkout> python compute/test/golden/gen_glued_hat_golden.py compute/test/golden/glued_hat_golden.txt"""
import sys

import numpy as np
import scipy.linalg as sla
from nxr.geom.glue import glue_mesh
from nxr.operators.faces import FaceOperators
from nxr.operators.intrinsic_delaunay import cotan_laplacian, lumped_mass, prepare


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


U, F0 = icosphere(3)
n1 = len(U)
R, GAP = 0.08, 0.2
P = np.vstack([U * R, U * R + [GAP, 0, 0]]).astype(np.float32).astype(np.float64)
F = np.vstack([F0, F0 + n1])
wall = np.zeros(2 * n1, bool)
wall[:n1] = P[:n1, 0] > 0.05
wall[n1:] = P[n1:, 0] - GAP < -0.05
nv = len(P)
rng = np.random.default_rng(7)
ops = FaceOperators(P, F)
ctr = np.vstack([np.zeros((n1, 3)), np.tile([GAP, 0, 0], (n1, 1))])
Un = np.vstack([U, U])
q = np.zeros((nv, 3, 3))
q[:, :, 0] = np.cross([0, 0, 1.0], P - ctr) / R + 0.3 * Un * np.sin(7 * Un[:, [0]])
q[:, :, 1] = np.cos(3 * Un[:, [1]]) * Un[:, [0]] * np.array([0.0, 1.0, 0.0]) + np.sin(4 * Un[:, [2]]) * np.array([1.0, 0.0, 0.0]) + 0.2 * Un
q[:, :, 2] = np.sin(2 * Un + 1.0) + 0.5 * np.cos(5 * Un[:, ::-1])
q = q.astype(np.float32).astype(np.float64)
dv, cu = ops.div_curl(ops.tangent(q))                                    # [nv, cols]

Vg, G, keep, bridge = glue_mesh(P, F, wall)
Vg = np.asarray(Vg, np.float64).astype(np.float32).astype(np.float64)   # the stored glued surface is float32
m = prepare(Vg, G, signposts=False)
L = cotan_laplacian(m).toarray()
M = lumped_mass(m)
lam, Phi = sla.eigh(L, np.diag(M))
lam = np.maximum(lam, 0)
nG = len(Vg)
xg = np.zeros((nG, 3))
for i in range(nG):
    if keep[i] >= 0:
        xg[i] = cu[keep[i]]
A = ops_area = 0.0
# a hemisphere's area: the native mesh's total vertex area over two (nxr: vertex_areas(V, F).sum() / 2)
a = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
A = np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0]), axis=1).sum() / 2 / 2
with open(sys.argv[1], "w") as out:
    out.write(f"nv {nv} nf {len(F)} nglued {nG} cols 3 area {float(A)!r}\n")
    out.write("positions " + " ".join(repr(float(v)) for v in P.ravel()) + "\n")
    out.write("faces " + " ".join(str(int(v)) for v in F.ravel()) + "\n")
    out.write("wall " + " ".join(str(int(v)) for v in wall) + "\n")
    out.write("glued_positions " + " ".join(repr(float(v)) for v in Vg.ravel()) + "\n")
    out.write("glued_faces " + " ".join(str(int(v)) for v in np.asarray(G).ravel()) + "\n")
    out.write("keep " + " ".join(str(int(v)) for v in keep) + "\n")
    out.write("q " + " ".join(repr(float(v)) for v in q.reshape(nv * 3, 3).ravel()) + "\n")          # [3 nv x 3 cols] row-major: row 3v + a
    out.write("curl " + " ".join(repr(float(v)) for v in cu.ravel()) + "\n")
    for level in (2, 3):
        lam_peak = 4 * np.pi * 2 ** (level - 0.5) / A
        s2 = 2 / lam_peak
        g = lam * s2 * np.exp(-lam * s2 / 2) / (2 / np.e)
        Y = Phi @ (g[:, None] * (Phi.T @ (M[:, None] * xg)))
        out.write(f"hat {level} {float(lam_peak)!r} " + " ".join(repr(float(v)) for v in Y.ravel()) + "\n")
        br = keep < 0
        print(f"level {level}: |hat| max {abs(Y).max():.3g}, on the bridge {abs(Y[br]).max():.3g} ({br.sum()} bridge vertices of {nG})")
print(f"wrote {sys.argv[1]}")
