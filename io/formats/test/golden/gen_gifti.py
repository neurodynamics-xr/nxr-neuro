#!/usr/bin/env python3
"""Synthetic GIFTI files and their goldens, made with nibabel (a reference for parity only: nxr-neuro reads and writes GIFTI natively).

Writes test/data/gifti_*.gii and test/golden/gifti_golden.json. Run:
  uv run --project <nxr> python test/golden/gen_gifti.py
With --check FILE it loads FILE with nibabel and prints its arrays (to check a file nxr-neuro wrote).
"""
import json
import pathlib
import sys
import numpy as np
import nibabel as nib
from nibabel import gifti

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
data_dir.mkdir(exist_ok=True)

if len(sys.argv) == 3 and sys.argv[1] == "--check":
    img = nib.load(sys.argv[2])
    for d in img.darrays:
        print(d.intent, d.datatype, d.data.shape, d.encoding, d.endian, d.data.ravel()[:6])
    print("labels", {k: v.label for k, v in img.labeltable.labels_as_dict.items()} if hasattr(img.labeltable, "labels_as_dict") else "")
    sys.exit(0)

rng = np.random.default_rng(20261117)
golden = {"nibabel": nib.__version__, "cases": {}}

# an octahedron: 6 vertices, 8 triangles
pts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32) * 30.0
pts += rng.normal(scale=0.01, size=pts.shape).astype(np.float32)
tris = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], dtype=np.int32)
func = rng.normal(size=6).astype(np.float32)
labels = np.array([0, 1, 1, 2, 2, 0], dtype=np.int32)
xf = np.eye(4)
xf[:3, 3] = [1.5, -2.5, 3.5]


def surf_image(encoding, endian):
    cs = gifti.GiftiCoordSystem(dataspace="NIFTI_XFORM_TALAIRACH", xformspace="NIFTI_XFORM_TALAIRACH", xform=xf)
    d_pts = gifti.GiftiDataArray(pts, intent="NIFTI_INTENT_POINTSET", datatype="NIFTI_TYPE_FLOAT32", encoding=encoding, endian=endian,
                                 coordsys=cs, meta=gifti.GiftiMetaData.from_dict({"AnatomicalStructurePrimary": "CortexLeft"}))
    d_tri = gifti.GiftiDataArray(tris, intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32", encoding=encoding, endian=endian,
                                 coordsys=None)
    return gifti.GiftiImage(darrays=[d_pts, d_tri], meta=gifti.GiftiMetaData.from_dict({"Description": "synthetic octahedron"}))


def func_image(encoding, endian):
    d = gifti.GiftiDataArray(func, intent="NIFTI_INTENT_SHAPE", datatype="NIFTI_TYPE_FLOAT32", encoding=encoding, endian=endian, coordsys=None,
                             meta=gifti.GiftiMetaData.from_dict({"Name": "synthetic shape"}))
    return gifti.GiftiImage(darrays=[d])


def label_image(encoding, endian):
    lt = gifti.GiftiLabelTable()
    for key, name, rgba in [(0, "???", (0.0, 0.0, 0.0, 0.0)), (1, "region_a", (1.0, 0.0, 0.0, 1.0)), (2, "region_b", (0.0, 0.5, 1.0, 1.0))]:
        lab = gifti.GiftiLabel(key=key, red=rgba[0], green=rgba[1], blue=rgba[2], alpha=rgba[3])
        lab.label = name
        lt.labels.append(lab)
    d = gifti.GiftiDataArray(labels, intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32", encoding=encoding, endian=endian, coordsys=None)
    return gifti.GiftiImage(darrays=[d], labeltable=lt)


NS = nib.nifti1
def intent_name(c): return NS.intent_codes.niistring[c]
def dtype_name(c): return NS.data_type_codes.niistring[c]
def xform_name(c): return NS.xform_codes.niistring[c]


def record(name, fname, encoding):
    img = nib.load(data_dir / fname)
    arrays = []
    for d in img.darrays:
        entry = {
            "intent": intent_name(d.intent), "datatype": dtype_name(d.datatype), "dims": [int(x) for x in d.data.shape],
            "encoding": encoding, "endian": "BigEndian" if d.endian == "big" or str(d.endian) in ("1", "GiftiEndian.BIG") else "LittleEndian",
            "data": [float(v) for v in d.data.ravel()],
            "metadata": dict(d.meta),
            "coordsys": None if d.coordsys is None else {"dataspace": xform_name(d.coordsys.dataspace), "xformspace": xform_name(d.coordsys.xformspace),
                                                           "xform": [float(v) for v in d.coordsys.xform.ravel()]},
        }
        arrays.append(entry)
    labels_ = {}
    for lab in img.labeltable.labels:
        labels_[str(lab.key)] = {"name": lab.label, "rgba": [lab.red, lab.green, lab.blue, lab.alpha]}
    golden["cases"][name] = {"file": fname, "metadata": dict(img.meta), "arrays": arrays, "labels": labels_}


for tag, enc in [("ascii", "ASCII"), ("b64", "B64BIN"), ("b64gz", "B64GZ")]:
    nib.save(surf_image(enc, "little"), data_dir / f"gifti_surf_{tag}.surf.gii")
    record(f"surf_{tag}", f"gifti_surf_{tag}.surf.gii", {"ascii": "ASCII", "b64": "Base64Binary", "b64gz": "GZipBase64Binary"}[tag])
# nibabel always writes its arrays little-endian, so the big-endian file is made from the little one: bytes swapped, Endian attribute changed
import base64
import xml.etree.ElementTree as ET
tree = ET.parse(data_dir / "gifti_surf_b64.surf.gii")
for da in tree.getroot().iter("DataArray"):
    width = 4
    raw = base64.b64decode(da.find("Data").text)
    swapped = np.frombuffer(raw, dtype=np.uint8).reshape(-1, width)[:, ::-1].tobytes()
    da.find("Data").text = base64.b64encode(swapped).decode()
    da.set("Endian", "BigEndian")
tree.write(data_dir / "gifti_surf_b64_be.surf.gii", encoding="UTF-8", xml_declaration=True)
record("surf_b64_be", "gifti_surf_b64_be.surf.gii", "Base64Binary")
nib.save(func_image("B64GZ", "little"), data_dir / "gifti_func.func.gii")
record("func", "gifti_func.func.gii", "GZipBase64Binary")
nib.save(label_image("ASCII", "little"), data_dir / "gifti_label.label.gii")
record("label", "gifti_label.label.gii", "ASCII")

(here / "gifti_golden.json").write_text(json.dumps(golden, indent=1))
print("wrote", len(golden["cases"]), "cases")
