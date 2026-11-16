"""Checks ladder_golden.txt (the Studio's ladder + the sums the generator writes) against nxr's OWN functions: geom.ladder.surface_ladder, measure.scalars.map_stats,
measure.vectors.vector_stats and lc_resultant. Run in nxr's environment:
    uv run --project <path to nxr> python compute/test/golden/check_ladder_nxr.py compute/test/golden/ladder_golden.txt
Prints the largest differences; exits 1 above 1e-12 (the Levi-Civita angles are the Studio's transport, read as given)."""
import sys
import numpy as np
from nxr.geom.ladder import surface_ladder, component_mesh
from nxr.geom.trees import Tree
from nxr.measure.scalars import map_stats
from nxr.measure.vectors import Frames, vector_stats, lc_resultant, LeviCivita

toks = open(sys.argv[1]).read().split()
pos = 0
def tok():
    global pos
    pos += 1
    return toks[pos - 1]
def mat(name):
    assert tok() == name, name
    r, c = int(tok()), int(tok())
    a = np.array([float(tok()) for _ in range(r * c)])
    return a.reshape(r, c)
worst = 0.0
for _ in range(2):
    assert tok() == "mesh"; tok(); n = int(tok())
    P, F, mass = mat("positions"), mat("faces").astype(int), mat("mass")[0]
    assert tok() == "levels"; K = int(tok()); tok()
    lad = surface_ladder(P, F)
    assert len(lad.labels) == K, (len(lad.labels), K)
    for k in range(K):
        mat(f"level{k}"); lab = mat(f"labels{k}")[0]; mat(f"areas{k}"); mat(f"counts{k}"); mat(f"centres{k}"); mat(f"adjacency{k}")
        assert (lad.labels[k] == lab).all(), f"labels differ at level {k}"
    x = mat("x")[0]
    for k in range(K):
        tree = Tree("subject", k, lad.labels[k].astype(np.int32), k)
        ref = map_stats(tree, mass, x)
        for key in ("w", "s1", "s2", "n", "min", "max"):
            g = mat(f"M{k}_{key}")[0]; m = ref[key][: len(g)]
            worst = max(worst, float(np.max(np.abs(np.where(np.isinf(g), 0, g - m)))))
    alpha = mat("alpha")[0]; e1, e2, nrm, v = mat("e1"), mat("e2"), mat("nrm"), mat("v")
    fr = Frames(e1, e2, nrm, np.zeros(n, bool))
    assert tok() == "lc"; L = int(tok())
    for k in range(1, L + 1):
        assert tok() == "level"; tok()
        rho = mat("rho")[0]; mat("omega")
        S = {key: mat(f"S_{key}")[0] for key in ("w", "c1", "c2", "cn", "c11", "c22", "c12", "mag")}
        re, im = mat("lc_re")[0], mat("lc_im")[0]
        tree = Tree("subject", k, lad.labels[k].astype(np.int32), k)
        ref = vector_stats(tree, mass, fr, v)
        for key, g in S.items():
            worst = max(worst, float(np.max(np.abs(ref[key][: len(g)] - g))))
        lc = LeviCivita((k,), {}, {k: rho}, {})
        z = lc_resultant(tree, mass, fr, lc, v, k)[: len(re)]
        worst = max(worst, float(np.max(np.abs(z.real - re))), float(np.max(np.abs(z.imag - im))))
print(f"nxr parity: labels identical at every level; max |diff| of map_stats, vector_stats, lc_resultant sums {worst:.2e}")
sys.exit(1 if worst > 1e-12 else 0)
