"""Writes group_tree_golden.txt: the reference of nxr::select::groupTreeFor, nxr's `geom.trees.Template.tree_for` (a subject's vertices on the group tree through the registered sphere:
each takes the code of the nearest template vertex, hemisphere by hemisphere) on SYNTHETIC random spheres and random template codes.
Run:  uv run --project <nxr checkout> python compute/test/golden/gen_group_tree_golden.py compute/test/golden/group_tree_golden.txt"""
import sys

import numpy as np
from nxr.geom.trees import Template

R = lambda v: repr(float(v))
rng = np.random.default_rng(11)
F = 6                                                    # the template's finest level
nt = {"lh": 300, "rh": 350}
sph = {h: rng.normal(size=(n, 3)) for h, n in nt.items()}
sph = {h: v / np.linalg.norm(v, axis=1, keepdims=True) for h, v in sph.items()}
codes = {h: rng.integers(0, 2 ** F, size=n).astype(np.int32) for h, n in nt.items()}
tpl = Template("synthetic", F, sph, codes)
ns = (120, 90)
S = np.vstack([rng.normal(size=(ns[0], 3)) * 2.5, rng.normal(size=(ns[1], 3)) * 0.4])          # not unit: the matcher normalises
structures = np.array([1] * ns[0] + [2] * ns[1])
out = open(sys.argv[1], "w")
out.write(f"finest {F}\n")
out.write("tpl_runs 0 %d %d %d\n" % (nt["lh"], nt["lh"], nt["rh"]))
out.write("tpl_sphere " + " ".join(R(x) for x in np.vstack([sph["lh"], sph["rh"]]).ravel()) + "\n")
# the template's codes with the hemisphere offset (what the stored ladder holds): h 2^F + t
out.write("tpl_codes " + " ".join(str(int(x)) for x in np.concatenate([codes["lh"], codes["rh"] + 2 ** F])) + "\n")
out.write("sub_runs 0 %d %d %d\n" % (ns[0], ns[0], ns[1]))
out.write("sub_sphere " + " ".join(R(x) for x in S.ravel()) + "\n")
for d in (0, 2, 4, 6):
    t = tpl.tree_for(S, structures, None, d)
    out.write(f"depth {d} median {R(t.meta['sphere_match_median'])} max {R(t.meta['sphere_match_max'])}\n")
    out.write("codes " + " ".join(str(int(x)) for x in t.codes) + "\n")
