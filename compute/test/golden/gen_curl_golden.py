"""Writes curl_golden.txt: the reference of nxr::signal::formOfFreeColumns (apply_kernel's `form: curl_tan | div_tan | normal`), nxr's `Cortex.readings` before its mode coordinates:
    q = K.reshape(nv, 3, -1);  dv, cu = ops.div_curl(ops.tangent(q));  normal = einsum(normals, q)      (nxr/detect/vortex.py; ops = nxr.operators.faces.FaceOperators)
on a SYNTHETIC jittered icosphere (radius 0.08 m, 642 vertices) with three planted free currents (columns): a rigid rotation about z (ω × r) plus a normal part, a
sink-source pattern (a gradient of a smooth scalar) plus a normal part, and a smooth mixture. The currents are rounded to float32 first (the Studio's precision; the same input
reaches both). Run:
  uv run --project <nxr checkout> python compute/test/golden/gen_curl_golden.py compute/test/golden/curl_golden.txt"""
import sys

import numpy as np
from nxr.operators.faces import FaceOperators


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


rng = np.random.default_rng(3)
U, F = icosphere(3)
V = (U * (0.08 * (1 + 0.02 * np.sin(5 * U[:, 0] + 3 * U[:, 1]) * np.cos(4 * U[:, 2])))[:, None]).astype(np.float32).astype(np.float64)       # a jittered sphere: normals are not radial
nv = len(V)
ops = FaceOperators(V, F)
z = np.array([0, 0, 1.0])
q = np.zeros((nv, 3, 3))
q[:, :, 0] = np.cross(z, V) / 0.08 + 0.3 * U * np.sin(7 * U[:, [0]])                                              # rotation about z, plus a normal part
q[:, :, 1] = (np.cos(3 * U[:, [1]]) * U[:, [0]] * np.array([0.0, 1.0, 0.0]) + np.sin(4 * U[:, [2]]) * np.array([1.0, 0.0, 0.0])) + 0.2 * U
q[:, :, 2] = np.sin(2 * U + 1.0) + 0.5 * np.cos(5 * U[:, ::-1])
q = q.astype(np.float32).astype(np.float64)
dv, cu = ops.div_curl(ops.tangent(q))
nrm = np.einsum("vk,vkc->vc", ops.normals, q)
with open(sys.argv[1], "w") as out:
    out.write(f"nv {nv} nf {len(F)} cols 3\n")
    out.write("positions " + " ".join(repr(float(v)) for v in V.ravel()) + "\n")
    out.write("faces " + " ".join(str(int(v)) for v in F.ravel()) + "\n")
    out.write("q " + " ".join(repr(float(v)) for v in q.reshape(nv * 3, 3).ravel()) + "\n")          # [3 nv × 3 cols] row-major: row 3v + a
    out.write("curl_tan " + " ".join(repr(float(v)) for v in cu.ravel()) + "\n")
    out.write("div_tan " + " ".join(repr(float(v)) for v in dv.ravel()) + "\n")
    out.write("normal " + " ".join(repr(float(v)) for v in nrm.ravel()) + "\n")
print(f"wrote {sys.argv[1]}  (|curl| max {abs(cu).max():.3g}, |div| max {abs(dv).max():.3g})")
