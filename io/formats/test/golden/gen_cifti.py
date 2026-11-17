#!/usr/bin/env python3
"""Synthetic CIFTI-2 files and their goldens, made with nibabel (a reference for parity only: nxr-neuro reads the format natively).
nibabel writes dense and parcellated files (dtseries, dscalar, dlabel, pscalar, ptseries) over invented surfaces' vertex subsets and a tiny volume;
nibabel reads them back for the goldens. No real data, no real subject.

Writes test/data/cifti/*.nii and test/golden/cifti_golden.json (and one raw .f32 of each file's matrix). Run:
  uv run --project <nxr> python test/golden/gen_cifti.py
"""
import json
import pathlib
import shutil
import struct
import warnings

import numpy as np
import nibabel as nib
from nibabel import cifti2

warnings.filterwarnings("ignore")
here = pathlib.Path(__file__).resolve().parent
root = here.parent / "data" / "cifti"
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
rng = np.random.default_rng(20261122)
golden = {"nibabel": nib.__version__, "cases": {}}

aff = np.array([[2.0, 0, 0, -10.0], [0, 2.0, 0, -12.0], [0, 0, 2.0, -6.0], [0, 0, 0, 1.0]])
mask = np.zeros((5, 6, 4), bool)
mask[1:3, 2:4, 1:3] = True
mask[4, 5, 3] = True
bm_l = cifti2.BrainModelAxis.from_surface(vertices=np.array([0, 2, 3, 7, 11]), nvertex=20, name="CIFTI_STRUCTURE_CORTEX_LEFT")
bm_r = cifti2.BrainModelAxis.from_surface(vertices=np.array([1, 4, 5, 9]), nvertex=18, name="CIFTI_STRUCTURE_CORTEX_RIGHT")
bm_v = cifti2.BrainModelAxis.from_mask(mask, name="CIFTI_STRUCTURE_THALAMUS_LEFT", affine=aff)
dense = bm_l + bm_r + bm_v     # 5 + 4 + 9 grayordinates
n_dense = len(dense)
series = cifti2.SeriesAxis(start=0.5, step=0.72, size=7, unit="SECOND")
scalars = cifti2.ScalarAxis(["alpha", "beta"], meta=[{"units": "au", "k": "v"}, {}])
labels = cifti2.LabelAxis(["parc1", "parc2"], [{0: ("???", (0.0, 0.0, 0.0, 0.0)), 1: ("Visual", (1.0, 0.0, 0.0, 1.0)), 2: ("Motor", (0.0, 1.0, 0.0, 1.0))},
                                              {0: ("???", (0.0, 0.0, 0.0, 0.0)), 5: ("Default", (0.0, 0.0, 1.0, 1.0))}])
sub = lambda *idx: dense[list(idx)]
parcels = cifti2.ParcelsAxis.from_brain_models([("P_left", dense[0:3]), ("P_mix", dense[3:12]), ("P_vol", dense[12:])])
parcels_alone = cifti2.ParcelsAxis.from_brain_models([("Pa", dense[0:5]), ("Pb", dense[5:9])])


def axis_json(ax):
    if isinstance(ax, cifti2.BrainModelAxis):
        out = {"kind": "brain_models", "structures": []}
        for name, slc, bm in ax.iter_structures():
            out["structures"].append({"name": name, "start": slc.start, "stop": len(ax) if slc.stop is None else slc.stop, "is_surface": bool((bm.vertex >= 0).all()),
                                      "vertex": bm.vertex.tolist(), "voxel": bm.voxel.tolist(), "nvertices": {k: int(v) for k, v in ax.nvertices.items()}.get(name)})
        out["volume_shape"] = None if ax.volume_shape is None else list(ax.volume_shape)
        out["affine"] = None if ax.affine is None else ax.affine.tolist()
        return out
    if isinstance(ax, cifti2.SeriesAxis):
        return {"kind": "series", "start": float(ax.start), "step": float(ax.step), "size": int(ax.size), "unit": ax.unit}
    if isinstance(ax, cifti2.ScalarAxis):
        return {"kind": "scalars", "names": [str(n) for n in ax.name], "meta": [dict(m) for m in ax.meta]}
    if isinstance(ax, cifti2.LabelAxis):
        return {"kind": "labels", "names": [str(n) for n in ax.name], "labels": [{str(k): [v[0], list(v[1])] for k, v in d.items()} for d in ax.label],
                "meta": [dict(m) for m in ax.meta]}
    if isinstance(ax, cifti2.ParcelsAxis):
        return {"kind": "parcels", "names": [str(n) for n in ax.name],
                "vertices": [{k: np.asarray(v).tolist() for k, v in vd.items()} for vd in ax.vertices],
                "voxels": [np.asarray(v).tolist() for v in ax.voxels], "nvertices": {k: int(v) for k, v in ax.nvertices.items()},
                "volume_shape": None if ax.volume_shape is None else list(ax.volume_shape), "affine": None if ax.affine is None else ax.affine.tolist()}
    raise TypeError(type(ax))


def make(name, ax0, ax1, dtype=np.float32):
    data = rng.normal(size=(len(ax0), len(ax1))).astype(dtype)
    img = cifti2.Cifti2Image(data, cifti2.Cifti2Header.from_axes((ax0, ax1)))
    path = root / f"{name}.nii"
    nib.save(img, path)
    back = nib.load(path)
    (here / f"cifti_{name}.f32").write_bytes(np.asarray(back.get_fdata(), dtype="<f8" if dtype == np.float64 else "<f4").tobytes())
    h = back.header
    golden["cases"][name] = {"file": f"{name}.nii", "shape": list(back.shape), "intent_code": int(back.nifti_header["intent_code"]), "intent_name": str(back.nifti_header["intent_name"].item().decode()) if hasattr(back.nifti_header["intent_name"].item(), "decode") else str(back.nifti_header["intent_name"]),
                             "axes": [axis_json(h.get_axis(0)), axis_json(h.get_axis(1))], "float64": dtype == np.float64,
                             "metadata": dict(h.matrix.metadata) if h.matrix.metadata is not None else {}}


make("dtseries", series, dense)
make("dscalar", scalars, dense)
make("dlabel", labels, dense)
make("pscalar", scalars, parcels)
make("ptseries", series, parcels)
make("pscalar_surf", scalars, parcels_alone)
make("dscalar64", scalars, dense, dtype=np.float64)

# refusals: a NIfTI-1 file, a file with no CIFTI extension, CIFTI-1 XML, a truncated file
src = (root / "dscalar.nii").read_bytes()
(root / "truncated.nii").write_bytes(src[:-200])
nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.float32), np.eye(4)), root / "nifti1.nii")
nib.save(nib.Nifti2Image(np.zeros((2, 2, 2), np.float32), np.eye(4)), root / "nifti2_plain.nii")
# CIFTI-1: the XML's Version attribute replaced (same length), so the file stays well formed
v1 = src.replace(b'Version="2.0"', b'Version="1.0"')
assert v1 != src
(root / "cifti1.nii").write_bytes(v1)
# a datatype this reader does not take: int16 (code 4) in the header's datatype and bitpix fields
b = bytearray(src)
struct.pack_into("<h", b, 12, 4)
struct.pack_into("<h", b, 14, 16)
(root / "int16.nii").write_bytes(bytes(b))

with open(here / "cifti_golden.json", "w") as f:
    json.dump(golden, f, separators=(",", ":"))
size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
print("cifti data", size, "bytes;", "golden", (here / "cifti_golden.json").stat().st_size)
