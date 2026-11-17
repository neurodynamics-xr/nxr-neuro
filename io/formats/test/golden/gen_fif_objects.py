#!/usr/bin/env python3
"""Synthetic FIF objects and their goldens, made with MNE-Python (a reference for parity only: nxr-neuro reads the format natively).
MNE writes the files from invented geometry (icosphere "hemispheres", a sphere head model, invented sensors); MNE reads them back for the goldens.
No real data, no real subject: every identifying value is invented.

Writes test/data/fif_obj_*.fif and test/golden/fif_objects_golden.json. Run:
  uv run --project <nxr> python test/golden/gen_fif_objects.py
"""
import datetime
import json
import pathlib
import tempfile
import warnings

import numpy as np
import mne
from mne._fiff.constants import FIFF
from mne._fiff.write import start_and_end_file, start_block, end_block, write_int, write_string
from mne.surface import _get_ico_surface
from scipy.sparse import triu

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
for old in data_dir.glob("fif_obj*"):
    old.unlink()
rng = np.random.default_rng(20261118)
golden = {"mne": mne.__version__}


def arr(x):
    return None if x is None else np.asarray(x).tolist()


def trans_json(t):
    return None if t is None else {"from": int(t["from"]), "to": int(t["to"]), "trans": np.asarray(t["trans"]).tolist()}


def sparse_json(m):
    if m is None:
        return None
    m = m.tocsr()
    return {"shape": list(m.shape), "data": m.data.astype(float).tolist(), "indices": m.indices.tolist(), "indptr": m.indptr.tolist()}


def space_json(s):
    d = {k: s.get(k) for k in ("type", "id", "coord_frame", "np", "ntri", "nuse", "nuse_tri", "subject_his_id", "dist_limit",
                               "mri_file", "mri_volume_name", "seg_name", "mri_width", "mri_height", "mri_depth")}
    for k in ("rr", "nn", "tris", "inuse", "vertno", "use_tris", "nearest", "nearest_dist", "shape"):
        v = s.get(k)
        d[k] = None if v is None else np.asarray(v).tolist()
    d["dist"] = sparse_json(triu(s["dist"], format="csr")) if s.get("dist") is not None else None
    d["interpolator"] = sparse_json(s.get("interpolator"))
    for k in ("src_mri_t", "vox_mri_t", "mri_ras_t"):
        d[k] = trans_json(s.get(k))
    nv = s.get("neighbor_vert")
    d["neighbor_vert"] = None if nv is None else [np.asarray(n).tolist() for n in nv]
    d["id"] = int(s["id"])
    if d["dist_limit"] is not None and not np.isfinite(d["dist_limit"]):
        d["dist_limit"] = "inf"   # JSON has no infinity
    return d


# ---- a fake subject: two icosphere "hemispheres" as white surfaces, with unit spheres for the spacing ----
tmp = pathlib.Path(tempfile.mkdtemp())
subj = tmp / "syn"
(subj / "surf").mkdir(parents=True)
ico = _get_ico_surface(2)   # 162 vertices on the unit sphere
for hemi, shift in (("lh", -0.02), ("rh", 0.02)):
    white = ico["rr"] * 0.04 + np.array([shift, 0, 0])
    mne.write_surface(str(subj / "surf" / f"{hemi}.white"), white * 1000, ico["tris"], overwrite=True)
    mne.write_surface(str(subj / "surf" / f"{hemi}.sphere"), ico["rr"] * 100, ico["tris"], overwrite=True)

src = mne.setup_source_space("syn", spacing="oct2", surface="white", subjects_dir=str(tmp), add_dist=True)
for s in src:
    s["subject_his_id"] = "syn"
src.save(data_dir / "fif_obj_surf-src.fif", overwrite=True)
src_rd = mne.read_source_spaces(data_dir / "fif_obj_surf-src.fif")
golden["src_surf"] = {"file": "fif_obj_surf-src.fif", "spaces": [space_json(s) for s in src_rd]}

# a volume source space (spherical grid, no MRI) and a discrete one
vol = mne.setup_volume_source_space(pos=30.0, sphere=(0.0, 0.0, 0.0, 0.07), mri=None, verbose=False)
vol.save(data_dir / "fif_obj_vol-src.fif", overwrite=True)
vol_rd = mne.read_source_spaces(data_dir / "fif_obj_vol-src.fif")
golden["src_vol"] = {"file": "fif_obj_vol-src.fif", "spaces": [space_json(s) for s in vol_rd]}
pos = dict(rr=rng.normal(size=(5, 3)) * 0.01, nn=np.tile([0.0, 0.0, 1.0], (5, 1)))
disc = mne.setup_volume_source_space(pos=pos, verbose=False)
disc.save(data_dir / "fif_obj_discrete-src.fif", overwrite=True)
disc_rd = mne.read_source_spaces(data_dir / "fif_obj_discrete-src.fif")
golden["src_discrete"] = {"file": "fif_obj_discrete-src.fif", "spaces": [space_json(s) for s in disc_rd]}

# ---- sensors: invented MEG magnetometers and EEG electrodes around the head ----
n_meg, n_eeg = 10, 8
names = [f"MEG{i:03d}" for i in range(n_meg)] + [f"EEG{i:03d}" for i in range(n_eeg)]
info = mne.create_info(names, 100.0, ["mag"] * n_meg + ["eeg"] * n_eeg)
with info._unlock():
    info["meas_date"] = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    for i, ch in enumerate(info["chs"]):
        th, ph = np.arccos(1 - 2 * (i + 0.5) / len(names)), 2.4 * i
        r = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        if ch["kind"] == FIFF.FIFFV_MEG_CH:
            z = r
            x = np.cross([0, 0, 1.0], z)
            x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
            y = np.cross(z, x)
            ch["loc"] = np.r_[r * 0.13, x, y, z]
        else:
            ch["loc"][:3] = r * 0.085
    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
    info["bads"] = ["EEG003"]
trans = mne.transforms.Transform("head", "mri", np.array([[1.0, 0, 0, 0.001], [0, 1.0, 0, -0.002], [0, 0, 1.0, 0.003], [0, 0, 0, 1.0]]))
mne.write_trans(data_dir / "fif_obj-trans.fif", trans)
golden["trans"] = {"file": "fif_obj-trans.fif", "trans": trans_json(mne.read_trans(data_dir / "fif_obj-trans.fif"))}

# ---- covariance: full (with a projector, bads, eigen decomposition, score) and diagonal ----
data = rng.normal(size=(len(names), 600)) * 1e-12
raw = mne.io.RawArray(data, info.copy())
vec = rng.normal(size=(1, n_eeg))
vec /= np.linalg.norm(vec)
raw.add_proj(mne.Projection(active=False, data=dict(col_names=names[n_meg:], row_names=None, data=vec, ncol=n_eeg, nrow=1),
                            desc="syn-proj", kind=FIFF.FIFFV_PROJ_ITEM_FIELD, explained_var=0.5))
cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, method="empirical", verbose=False)
ev, evec = np.linalg.eigh(cov["data"])
cov["eig"], cov["eigvec"] = ev, evec.T
cov["loglik"] = -123.5
mne.write_cov(data_dir / "fif_obj_full-cov.fif", cov, overwrite=True)
cov_diag = mne.make_ad_hoc_cov(info)
mne.write_cov(data_dir / "fif_obj_diag-cov.fif", cov_diag, overwrite=True)


def cov_json(c):
    return {"kind": int(c["kind"]), "diag": bool(c["diag"]), "dim": int(c["dim"]), "nfree": int(c["nfree"]), "names": list(c["names"]),
            "data": np.asarray(c["data"]).tolist(), "bads": list(c["bads"]), "method": c.get("method"),
            "score": c.get("loglik"), "eig": arr(c.get("eig")), "eigvec": arr(c.get("eigvec")),
            "projs": [{"desc": p["desc"], "kind": int(p["kind"]), "active": bool(p["active"]), "nvec": int(p["data"]["nrow"]),
                       "channels": list(p["data"]["col_names"]), "vectors": np.asarray(p["data"]["data"]).tolist(),
                       "explained_var": p["explained_var"]} for p in c["projs"]]}


golden["cov_full"] = {"file": "fif_obj_full-cov.fif", **cov_json(mne.read_cov(data_dir / "fif_obj_full-cov.fif"))}
golden["cov_diag"] = {"file": "fif_obj_diag-cov.fif", **cov_json(mne.read_cov(data_dir / "fif_obj_diag-cov.fif"))}

# ---- forward solutions on a sphere model: free orientation MEG+EEG, fixed orientation MEG ----
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.09, verbose=False)
src_mri = mne.setup_source_space("syn", spacing="oct2", surface="white", subjects_dir=str(tmp), add_dist=False)
fwd = mne.make_forward_solution(info, trans=trans, src=src_mri, bem=sphere, meg=True, eeg=True, mindist=0.0, verbose=False)
mne.write_forward_solution(data_dir / "fif_obj_free-fwd.fif", fwd, overwrite=True)
fwd_meg = mne.make_forward_solution(info, trans=trans, src=src_mri, bem=sphere, meg=True, eeg=False, mindist=0.0, verbose=False)
fwd_fixed = mne.convert_forward_solution(fwd_meg, surf_ori=True, force_fixed=True, verbose=False)
mne.write_forward_solution(data_dir / "fif_obj_fixed-fwd.fif", fwd_fixed, overwrite=True)


def fwd_json(path):
    f = mne.read_forward_solution(path, verbose=False)
    return {"file": pathlib.Path(path).name, "source_ori": int(f["source_ori"]), "coord_frame": int(f["coord_frame"]), "nsource": int(f["nsource"]),
            "nchan": int(f["nchan"]), "sol": np.asarray(f["sol"]["data"]).tolist(), "row_names": list(f["sol"]["row_names"]),
            "col_names": list(f["sol"]["col_names"]), "ch_names": list(f["info"]["ch_names"]), "bads": list(f["info"]["bads"]),
            "mri_head_t": trans_json(f["mri_head_t"]), "dev_head_t": trans_json(f["info"]["dev_head_t"]),
            "mri_file": f["info"]["mri_file"], "meas_file": f["info"]["meas_file"], "nuse": int(sum(s["nuse"] for s in f["src"])),
            "ch_kinds": [int(c["kind"]) for c in f["info"]["chs"]], "ch_loc": [[None if np.isnan(x) else x for x in np.asarray(c["loc"]).tolist()] for c in f["info"]["chs"]]}   # null: NaN (no reference position)


golden["fwd_free"] = fwd_json(data_dir / "fif_obj_free-fwd.fif")
golden["fwd_fixed"] = fwd_json(data_dir / "fif_obj_fixed-fwd.fif")

# ---- an inverse operator from the free forward and the full covariance ----
inv = mne.minimum_norm.make_inverse_operator(info, fwd, mne.read_cov(data_dir / "fif_obj_full-cov.fif"), loose=0.2, depth=0.8, verbose=False)
mne.minimum_norm.write_inverse_operator(data_dir / "fif_obj-inv.fif", inv, overwrite=True)
iv = mne.minimum_norm.read_inverse_operator(data_dir / "fif_obj-inv.fif", verbose=False)
golden["inv"] = {"file": "fif_obj-inv.fif", "methods": int(iv["methods"]), "source_ori": int(iv["source_ori"]), "nsource": int(iv["nsource"]),
                 "nchan": int(iv["nchan"]), "coord_frame": int(np.asarray(iv["coord_frame"]).ravel()[0]), "units": iv["units"], "source_nn": np.asarray(iv["source_nn"]).tolist(),
                 "sing": np.asarray(iv["sing"]).tolist(), "eigen_leads": np.asarray(iv["eigen_leads"]["data"]).tolist(),
                 "eigen_leads_weighted": bool(iv["eigen_leads_weighted"]), "eigen_fields": np.asarray(iv["eigen_fields"]["data"]).tolist(),
                 "eigen_fields_rows": list(iv["eigen_fields"]["row_names"]), "noise_cov": cov_json(iv["noise_cov"]),
                 "source_cov": cov_json(iv["source_cov"]),
                 "orient_prior": None if iv["orient_prior"] is None else cov_json(iv["orient_prior"]),
                 "depth_prior": None if iv["depth_prior"] is None else cov_json(iv["depth_prior"]),
                 "n_projs": len(iv["projs"]), "mri_head_t": trans_json(iv["mri_head_t"]), "ch_names": list(iv["info"]["ch_names"]),
                 "mri_file": iv["info"]["mri_file"], "meas_file": iv["info"]["meas_file"]}

# ---- events (-eve.fif), with and without an event description ----
events = np.array([[100, 0, 1], [250, 1, 2], [400, 2, 1], [655, 1, 3], [900, 3, 2]], dtype=np.int32)
mne.write_events(data_dir / "fif_obj-eve.fif", events)
ev_rd = mne.read_events(data_dir / "fif_obj-eve.fif")
golden["events"] = {"file": "fif_obj-eve.fif", "events": ev_rd.tolist(), "event_id": None}
with start_and_end_file(data_dir / "fif_obj_named-eve.fif") as fid:
    start_block(fid, FIFF.FIFFB_MNE_EVENTS)
    write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, events.T)
    write_string(fid, FIFF.FIFF_DESCRIPTION, "left:1;right:2;a:b:3")
    end_block(fid, FIFF.FIFFB_MNE_EVENTS)
ev_rd, ev_id = mne.event.read_events(data_dir / "fif_obj_named-eve.fif", return_event_id=True)
golden["events_named"] = {"file": "fif_obj_named-eve.fif", "events": ev_rd.tolist(), "event_id": ev_id}

# ---- annotations (-annot.fif): with an origin time, channel names and extras; and relative to the first sample ----
orig = datetime.datetime(2026, 1, 2, 3, 4, 5, 678900, tzinfo=datetime.timezone.utc)
ann = mne.Annotations(onset=[1.5, 3.25, 10.0], duration=[0.5, 0.0, 2.0], description=["BAD_blink", "stim: a", "rest"], orig_time=orig,
                      ch_names=[["EEG001", "EEG002"], [], ["MEG000"]], extras=[{"score": 3, "label": "x"}, None, {"flag": True}])
ann.save(data_dir / "fif_obj-annot.fif", overwrite=True)
ann_rel = mne.Annotations(onset=[0.5, 2.0], duration=[1.0, 0.25], description=["a", "b"], orig_time=None)
ann_rel.save(data_dir / "fif_obj_rel-annot.fif", overwrite=True)


def ann_json(path):
    a = mne.read_annotations(path)
    return {"file": pathlib.Path(path).name, "onset": a.onset.tolist(), "duration": a.duration.tolist(), "description": list(a.description),
            "orig_time": None if a.orig_time is None else a.orig_time.timestamp(),
            "ch_names": [list(c) for c in a.ch_names], "extras": [None if not e else dict(e) for e in a.extras]}


golden["annot"] = ann_json(data_dir / "fif_obj-annot.fif")
golden["annot_rel"] = ann_json(data_dir / "fif_obj_rel-annot.fif")

with open(here / "fif_objects_golden.json", "w") as f:
    json.dump(golden, f, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
for p in sorted(data_dir.glob("fif_obj*")):
    print(p.name, p.stat().st_size)
print("golden", (here / "fif_objects_golden.json").stat().st_size)
