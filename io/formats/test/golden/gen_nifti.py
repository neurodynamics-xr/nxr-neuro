#!/usr/bin/env python3
"""Synthetic NIfTI files and their goldens, made with nibabel (a reference for parity only: nxr-neuro reads and writes NIfTI natively).

Writes test/data/nifti_*.nii* and test/golden/nifti_golden.json. Run:
  uv run --project <nxr> python test/golden/gen_nifti.py
"""
import gzip
import json
import pathlib
import numpy as np
import nibabel as nib

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
data_dir.mkdir(exist_ok=True)
rng = np.random.default_rng(20261117)
golden = {"nibabel": nib.__version__, "cases": {}}


def record(name, fname, img, extra=None):
    h = img.header
    arr = np.asanyarray(img.dataobj)
    zooms = [float(z) for z in h.get_zooms()]
    q = h.get_qform(coded=True)
    s = h.get_sform(coded=True)
    case = {
        "file": fname,
        "version": 2 if isinstance(h, nib.Nifti2Header) else 1,
        "shape": [int(x) for x in arr.shape],
        "zooms": zooms,
        "datatype": int(h["datatype"]),
        "scl_slope": None if not np.isfinite(h["scl_slope"]) else float(h["scl_slope"]),
        "scl_inter": None if not np.isfinite(h["scl_inter"]) else float(h["scl_inter"]),
        "qform_code": int(h["qform_code"]),
        "sform_code": int(h["sform_code"]),
        "xyzt_units": int(h["xyzt_units"]),
        "descrip": h["descrip"].tobytes().split(b"\0")[0].decode(),
        "best_affine": [float(v) for v in img.affine[:3, :].ravel()],   # nibabel's best affine (sform, else qform)
        "qform_affine": None if q[1] == 0 else [float(v) for v in q[0][:3, :].ravel()],
        "data": [float(v) for v in arr.ravel(order="F")],               # x fastest, as stored
    }
    if extra:
        case.update(extra)
    golden["cases"][name] = case


shape = (5, 4, 3)
base = rng.normal(size=shape).astype(np.float32)
aff = np.array([[2.0, 0, 0, -5.0], [0, 2.5, 0, -6.0], [0, 0, 3.0, -7.0], [0, 0, 0, 1]])

# 1. NIfTI-1, float32, sform
img = nib.Nifti1Image(base, aff)
img.header["descrip"] = b"synthetic float32 with sform"
img.header.set_xyzt_units("mm", "sec")
nib.save(img, data_dir / "nifti_f32_sform.nii")
record("f32_sform", "nifti_f32_sform.nii", nib.load(data_dir / "nifti_f32_sform.nii"))

# 2. NIfTI-1, int16 with a scaling, qform only (rotated, left-handed), sform code 0
th = 0.3
rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
aff_q = np.eye(4)
aff_q[:3, :3] = rot * np.array([1.5, 1.5, 2.0])
aff_q[:3, 3] = [10.0, -20.0, 30.0]
ints = rng.integers(-1000, 1000, size=shape).astype(np.int16)
img = nib.Nifti1Image(ints, None)
img.header.set_qform(aff_q, code=1)
img.header.set_sform(None, code=0)
img.header.set_slope_inter(0.5, 3.0)
img.header.set_xyzt_units("mm")
nib.save(img, data_dir / "nifti_i16_qform_scaled.nii")
record("i16_qform_scaled", "nifti_i16_qform_scaled.nii", nib.load(data_dir / "nifti_i16_qform_scaled.nii"))

# 3. NIfTI-1 uint8 gzipped, 4D
u8 = rng.integers(0, 255, size=(3, 3, 2, 4)).astype(np.uint8)
img = nib.Nifti1Image(u8, aff)
nib.save(img, data_dir / "nifti_u8_4d.nii.gz")
record("u8_4d_gz", "nifti_u8_4d.nii.gz", nib.load(data_dir / "nifti_u8_4d.nii.gz"))

# 4. NIfTI-2, float64
f64 = rng.normal(size=shape)
img = nib.Nifti2Image(f64, aff)
img.header["descrip"] = b"nifti2 float64"
nib.save(img, data_dir / "nifti2_f64.nii")
record("nifti2_f64", "nifti2_f64.nii", nib.load(data_dir / "nifti2_f64.nii"))

# 5. NIfTI-1 int32 (read as float64)
i32 = rng.integers(-2**30, 2**30, size=shape).astype(np.int32)
nib.save(nib.Nifti1Image(i32, aff), data_dir / "nifti_i32.nii")
record("i32", "nifti_i32.nii", nib.load(data_dir / "nifti_i32.nii"))

# 6. big-endian NIfTI-1 float32
img = nib.Nifti1Image(base.astype(">f4"), aff, header=nib.Nifti1Header(endianness=">"))
nib.save(img, data_dir / "nifti_f32_be.nii")
record("f32_be", "nifti_f32_be.nii", nib.load(data_dir / "nifti_f32_be.nii"), {"big_endian": True})

# 7. a pair (.hdr + .img)
img = nib.Nifti1Pair(base, aff)
nib.save(img, data_dir / "nifti_pair.img")
record("pair", "nifti_pair.img", nib.load(data_dir / "nifti_pair.img"))

# 8. no spatial form at all (codes 0): the scaling-only map of the pixdims
img = nib.Nifti1Image(base, None)
img.header.set_zooms((1.5, 2.0, 2.5))
img.header.set_qform(None, code=0)
img.header.set_sform(None, code=0)
nib.save(img, data_dir / "nifti_noform.nii")
c = nib.load(data_dir / "nifti_noform.nii")
record("noform", "nifti_noform.nii", c, {"best_affine": [1.5, 0, 0, 0, 0, 2.0, 0, 0, 0, 0, 2.5, 0]})

# refusals: a complex volume, a truncated file, an Analyze-like header
nib.save(nib.Nifti1Image(base.astype(np.complex64), aff), data_dir / "nifti_complex64.nii")
raw = (data_dir / "nifti_f32_sform.nii").read_bytes()
(data_dir / "nifti_truncated.nii").write_bytes(raw[:-40])
(data_dir / "not_nifti.nii").write_bytes(b"\x00" * 400)

(here / "nifti_golden.json").write_text(json.dumps(golden, indent=1))
print("wrote", len(golden["cases"]), "cases")
