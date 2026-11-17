#!/usr/bin/env python3
"""Synthetic FIF raw files and their goldens, made with MNE-Python (a reference for parity only: nxr-neuro reads the format natively).
MNE writes the files (`mne.io.RawArray(...).save`) from generated signals and an invented `Info`; MNE reads them back for the goldens.
No real data, no real subject: every identifying value below is invented.

Writes test/data/fif_*.fif, test/golden/fif_golden.json, and per case a raw `.f32` (or `.f64` for double files) of the data as MNE reads it
(channel after channel, samples in order, no projector applied). Run:
  uv run --project <nxr> python test/golden/gen_fif.py
"""
import datetime
import json
import pathlib
import warnings

import numpy as np
import mne
from mne._fiff.constants import FIFF
from mne._fiff._digitization import _format_dig_points
from mne.utils.numerics import _dt_to_stamp
from mne.annotations import _read_annotations_fif
from mne._fiff.open import fiff_open

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
data_dir.mkdir(exist_ok=True)
for old in data_dir.glob("fif_*"):
    old.unlink()
import mne.io.base
# MNE reserves 1 MB at the end of a part for its next-file tags; the synthetic splits shrink it so a part holds a few kB (the bytes written are the same)
mne.io.base._NEXT_FILE_BUFFER = 2000
rng = np.random.default_rng(20261117)
golden = {"mne": mne.__version__, "cases": {}}

SFREQ = 100.0
MEAS_DATE = datetime.datetime(2026, 1, 2, 3, 4, 5, 678900, tzinfo=datetime.timezone.utc)


def make_info(kinds, sfreq=SFREQ, meas_date=MEAS_DATE):
    names = [f"{k.upper()[:3]}{i:03d}" for i, k in enumerate(kinds)]
    info = mne.create_info(names, sfreq, kinds)
    with info._unlock():
        info["meas_date"] = meas_date
    return info


def full_info(info):
    """Everything the reader must parse: device and head transforms, dig, projectors, subject, bads, names, filters."""
    info = info.copy()
    with info._unlock():
        for i, ch in enumerate(info["chs"]):
            if ch["kind"] == FIFF.FIFFV_MEG_CH:
                ang = 0.1 * i
                rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
                ch["loc"] = np.r_[np.array([0.01 * i, -0.02, 0.05 + 0.001 * i]), rot.T.ravel()]
            elif ch["kind"] == FIFF.FIFFV_EEG_CH:
                ch["loc"][:3] = [0.02 * i, 0.03, 0.07]
        info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.array(
            [[0.99, -0.1, 0.0, 0.001], [0.1, 0.99, 0.0, -0.002], [0.0, 0.0, 1.0, 0.04], [0, 0, 0, 1]]))
        info["line_freq"] = 50.0
        info["highpass"] = 0.5
        info["lowpass"] = 40.0
        info["experimenter"] = "Invented Experimenter"
        info["description"] = "synthetic test recording"
        info["proj_name"] = "synthetic project"
        info["subject_info"] = dict(id=7, his_id="SYN-0007", last_name="Invented", first_name="Test", middle_name="Q",
                                    birthday=datetime.date(1980, 2, 29), sex=2, hand=1, weight=70.5, height=1.75)
        info["device_info"] = dict(type="synthetic", model="s1", serial="0000", site="nowhere")
        info["helium_info"] = dict(he_level_raw=82.5, helium_level=90.0, orig_file_guid="synthetic-guid", meas_date=MEAS_DATE)
        info["gantry_angle"] = 68
        info["utc_offset"] = "+0100"
    dig = [
        dict(kind=FIFF.FIFFV_POINT_CARDINAL, ident=FIFF.FIFFV_POINT_LPA, r=np.array([-0.08, 0.0, 0.0], np.float32), coord_frame=FIFF.FIFFV_COORD_HEAD),
        dict(kind=FIFF.FIFFV_POINT_CARDINAL, ident=FIFF.FIFFV_POINT_NASION, r=np.array([0.0, 0.1, 0.0], np.float32), coord_frame=FIFF.FIFFV_COORD_HEAD),
        dict(kind=FIFF.FIFFV_POINT_CARDINAL, ident=FIFF.FIFFV_POINT_RPA, r=np.array([0.08, 0.0, 0.0], np.float32), coord_frame=FIFF.FIFFV_COORD_HEAD),
        dict(kind=FIFF.FIFFV_POINT_HPI, ident=1, r=np.array([0.03, 0.05, 0.01], np.float32), coord_frame=FIFF.FIFFV_COORD_HEAD),
        dict(kind=FIFF.FIFFV_POINT_EEG, ident=1, r=np.array([0.02, 0.03, 0.07], np.float32), coord_frame=FIFF.FIFFV_COORD_HEAD),
        dict(kind=FIFF.FIFFV_POINT_EXTRA, ident=1, r=np.array([0.0, 0.05, 0.09], np.float32), coord_frame=FIFF.FIFFV_COORD_HEAD),
    ]
    with info._unlock():
        info["dig"] = _format_dig_points(dig)
    return info


def proj(info, name, active, vec_seed):
    chs = [c for c in info["ch_names"] if c.startswith("EEG")]
    v = np.random.default_rng(vec_seed).normal(size=(1, len(chs)))
    v /= np.linalg.norm(v)
    return mne.Projection(active=active, data=dict(col_names=chs, row_names=None, data=v, ncol=len(chs), nrow=1),
                          desc=name, kind=FIFF.FIFFV_PROJ_ITEM_FIELD, explained_var=0.25 if active else None)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return {"re": x.real.tolist(), "im": x.imag.tolist()}
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, mne.transforms.Transform):
        return {"from": int(x["from"]), "to": int(x["to"]), "trans": x["trans"].tolist()}
    if isinstance(x, (datetime.date, datetime.datetime)):
        return x.isoformat()
    return x


def dump_info(info):
    chs = [{k: (jsonable(v) if k != "loc" else [None if np.isnan(a) else float(a) for a in v]) for k, v in c.items()} for c in info["chs"]]
    out = dict(nchan=info["nchan"], sfreq=info["sfreq"], highpass=info["highpass"], lowpass=info["lowpass"], line_freq=info["line_freq"],
               bads=list(info["bads"]), ch_names=list(info["ch_names"]), chs=chs,
               meas_date=(None if info["meas_date"] is None else info["meas_date"].isoformat()),
               meas_date_stamp=(None if info["meas_date"] is None else [int(v) for v in _dt_to_stamp(info["meas_date"])]),
               utc_offset=info.get("utc_offset"), experimenter=info["experimenter"], description=info["description"],
               proj_id=info["proj_id"], proj_name=info["proj_name"], gantry_angle=info.get("gantry_angle"),
               custom_ref_applied=int(info["custom_ref_applied"]), subject_info=jsonable(info["subject_info"]),
               device_info=jsonable(info.get("device_info")), helium_info=jsonable(info.get("helium_info")),
               dev_head_t=jsonable(info["dev_head_t"]), ctf_head_t=jsonable(info["ctf_head_t"]),
               dig=jsonable(info["dig"]) if info["dig"] is not None else None,
               projs=[dict(desc=p["desc"], kind=int(p["kind"]), active=bool(p["active"]), nvec=p["data"]["nrow"], col_names=p["data"]["col_names"],
                           data=p["data"]["data"].tolist(), explained_var=p["explained_var"]) for p in info["projs"]],
               comps=[dict(ctfkind=int(c["ctfkind"]), kind=int(c["kind"]), save_calibrated=bool(c["save_calibrated"]),
                           row_names=c["data"]["row_names"], col_names=c["data"]["col_names"], data=c["data"]["data"].tolist(),
                           rowcals=c["rowcals"].tolist(), colcals=c["colcals"].tolist()) for c in info["comps"]],
               meas_id_secs=None if info["meas_id"] is None else int(info["meas_id"]["secs"]),
               file_id=None if info["file_id"] is None else dict(secs=int(info["file_id"]["secs"]), usecs=int(info["file_id"]["usecs"])))
    return out


def record(name, path, fmt_dtype="<f4", extra=None, skip_note=None):
    """Read `path` with MNE (every part of a split file) and write the golden: header, annotations, gaps, data."""
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    data = raw.get_data(reject_by_annotation=None)
    suffix = ".f64" if fmt_dtype == "<f8" else ".f32"
    if np.iscomplexobj(data):
        suffix = ".c128"
        out = np.stack([data.real, data.imag], axis=-1).astype("<f8")
    else:
        out = data.astype(fmt_dtype)
    (here / f"fif_{name}{suffix}").write_bytes(out.tobytes())
    # the annotations the FILE holds (not the BAD_ACQ_SKIP ones MNE adds for skips), onsets relative to the first sample
    first_time = raw.first_samp / raw.info["sfreq"]
    ff, tree, _ = fiff_open(path)
    with ff as fid:
        ann = _read_annotations_fif(fid, tree)
    rel = [] if ann is None else (np.asarray(ann.onset) - first_time).tolist()
    keep = [] if ann is None else list(range(len(ann.description)))
    gaps = []
    for ex in raw._raw_extras:
        b = ex["bounds"]
        for k, e in enumerate(ex["ent"]):
            if e is None:
                gaps.append((int(b[k] - raw._raw_extras[0]["bounds"][0]), int(b[k + 1] - b[k])))
    case = dict(file=path.name, data=f"fif_{name}{suffix}", n_times=int(raw.n_times), first_samp=int(raw.first_samp), sfreq=raw.info["sfreq"],
                orig_format=raw.orig_format, info=dump_info(raw.info),
                annotations=[dict(onset=float(rel[i]), duration=float(ann.duration[i]), label=str(ann.description[i])) for i in keep],
                gaps=[dict(first=a, count=b) for a, b in gaps], n_parts=len(raw.filenames))
    if extra:
        case.update(extra)
    golden["cases"][name] = case
    return raw


def save(raw, name, **kw):
    p = data_dir / f"fif_{name}_raw.fif"
    raw.save(p, overwrite=True, verbose="ERROR", **kw)
    return p


def int_signal(info, n, scale_by_kind=None):
    """Integers times each channel's cal*range, so the 16-bit storage is exact."""
    nch = len(info["chs"])
    ints = rng.integers(-2000, 2000, size=(nch, n))
    cals = np.array([c["cal"] * c["range"] for c in info["chs"]])
    return ints * cals[:, None]


# 1. the full header: every kind of channel, float32, a non-zero first sample, bads, dig, transforms, projectors, subject, annotations
kinds = ["mag", "mag", "grad", "grad", "eeg", "eeg", "eeg", "eog", "ecg", "stim"]
info = full_info(make_info(kinds))
with info._unlock():
    for ch in info["chs"]:
        if ch["kind"] == FIFF.FIFFV_MEG_CH:
            ch["cal"], ch["range"] = 2.0e-3, 1.0
    info["bads"] = ["MAG001", "EEG006"]
    info["projs"] = [proj(info, "synthetic-inactive", False, 1), proj(info, "synthetic-second", False, 2)]
n = 600
sig = rng.normal(scale=1e-6, size=(len(kinds), n))
sig[:2] *= 1e-6
sig[-1] = (np.arange(n) // 50) % 4
raw = mne.io.RawArray(sig, info, first_samp=250)
raw.set_annotations(mne.Annotations([3.0, 4.5, 6.75], [0.25, 0.0, 1.0], ["blink", "stim onset", "BAD_segment"], orig_time=MEAS_DATE))
p = save(raw, "basic", buffer_size_sec=1.0)
record("basic", p)

# 2. the data types: int16 (cal and range at work), int32, float64
info2 = make_info(["eeg", "eeg", "mag", "mag"])
with info2._unlock():
    for ch in info2["chs"]:
        ch["range"], ch["cal"] = 0.5, 2e-3
raw2 = mne.io.RawArray(int_signal(info2, 300), info2)
record("short", save(raw2, "short", fmt="short", buffer_size_sec=0.7))
ints32 = rng.integers(-2**30, 2**30, size=(4, 300)).astype(float)
raw3 = mne.io.RawArray(ints32 * np.array([c["cal"] * c["range"] for c in info2["chs"]])[:, None], info2.copy())
record("int", save(raw3, "int", fmt="int", buffer_size_sec=0.7))
info4 = make_info(["eeg", "eeg", "mag"])
raw4 = mne.io.RawArray(rng.normal(scale=1e-5, size=(3, 300)) + 1.0 / 3.0, info4)
record("double", save(raw4, "double", fmt="double", buffer_size_sec=0.7), fmt_dtype="<f8")
def relabel_buffers_complex(src, dst, new_type):
    """MNE-Python 1.13 writes complex raw data wrongly (RawArray(...).save of a complex array stores real bytes under a complex tag), so the
    complex files are made from valid real ones: the same data buffers, their tag type changed to complex float / complex double, which is
    what a complex buffer is on disk (nchan x nsamp complex values, sample-major). MNE then reads them back for the golden."""
    b = bytearray(src.read_bytes())
    pos, n = 0, 0
    while pos + 16 <= len(b):
        kind = int.from_bytes(b[pos:pos + 4], "big", signed=True)
        size = int.from_bytes(b[pos + 8:pos + 12], "big", signed=True)
        nxt = int.from_bytes(b[pos + 12:pos + 16], "big", signed=True)
        if kind == FIFF.FIFF_DATA_BUFFER:
            b[pos + 4:pos + 8] = new_type.to_bytes(4, "big")
            n += 1
        if nxt == 0:
            pos += 16 + size
        elif nxt > 0:
            pos = nxt
        else:
            break
    dst.write_bytes(bytes(b))
    return n


rawc = mne.io.RawArray(rng.normal(scale=1e-6, size=(3, 400)), make_info(["eeg", "eeg", "mag"]))
tmp32 = save(rawc, "tmp_c32", fmt="single", buffer_size_sec=1.0)
relabel_buffers_complex(tmp32, data_dir / "fif_complex_raw.fif", FIFF.FIFFT_COMPLEX_FLOAT)
tmp32.unlink()
record("complex", data_dir / "fif_complex_raw.fif", fmt_dtype="<f8")
tmp64 = save(rawc, "tmp_c64", fmt="double", buffer_size_sec=1.0)
relabel_buffers_complex(tmp64, data_dir / "fif_complexd_raw.fif", FIFF.FIFFT_COMPLEX_DOUBLE)
tmp64.unlink()
record("complexd", data_dir / "fif_complexd_raw.fif", fmt_dtype="<f8")

# 3. a split file (the second and third parts continue the first)
info6 = make_info(["eeg", "eeg", "eeg", "mag"])
raw6 = mne.io.RawArray(rng.normal(scale=1e-5, size=(4, 900)), info6, first_samp=40)
raw6.set_annotations(mne.Annotations([1.0, 5.0, 8.0], [0.5, 0.0, 0.5], ["early", "middle", "late"], orig_time=MEAS_DATE))
p = data_dir / "fif_split_raw.fif"
raw6.save(p, overwrite=True, split_size=9000, buffer_size_sec=0.5, verbose="ERROR")
record("split", p)

# 4. acquisition skips: buffers the file does not hold are zeros, read as gaps
info7 = make_info(["eeg", "eeg", "mag"])
sig7 = rng.normal(scale=1e-5, size=(3, 800))
sig7[:, 200:400] = 0.0
raw7 = mne.io.RawArray(sig7, info7)
raw7.set_annotations(mne.Annotations([2.0, 5.0], [2.0, 1.0], ["BAD_ACQ_SKIP", "keep"], orig_time=MEAS_DATE))
record("skip", save(raw7, "skip", buffer_size_sec=1.0))

# 5. no measurement date, a non-zero first sample, an active projector (header only: MNE applies it to the samples it reads)
info8 = full_info(make_info(["eeg", "eeg", "eeg", "mag"], meas_date=None))
with info8._unlock():
    info8["projs"] = [proj(info8, "synthetic-active", True, 3)]
    info8["subject_info"] = None
    info8["description"] = None
    info8["helium_info"] = None
raw8 = mne.io.RawArray(rng.normal(scale=1e-5, size=(4, 250)), info8, first_samp=500)
raw8.set_annotations(mne.Annotations([1.0], [1.0], ["after start"]))
record("nodate_activeproj", save(raw8, "nodate_activeproj", buffer_size_sec=1.0), extra=dict(data_has_projector=True))

# 6. CTF-style compensation: reference channels, a grade, a compensation matrix
info9 = make_info(["mag", "mag", "mag", "mag"])
with info9._unlock():
    names = ["MEG001", "MEG002", "REF001", "REF002"]
    for ch, nm in zip(info9["chs"], names):
        ch["ch_name"] = nm
    for ch in info9["chs"][:2]:
        ch["coil_type"] = FIFF.FIFFV_COIL_CTF_GRAD | (3 << 16)
    for ch in info9["chs"][2:]:
        ch["kind"] = FIFF.FIFFV_REF_MEG_CH
        ch["coil_type"] = FIFF.FIFFV_COIL_CTF_REF_MAG
    info9["comps"] = [dict(ctfkind=int("47334252", 16), kind=3, save_calibrated=True,
                           data=dict(nrow=2, ncol=2, row_names=["MEG001", "MEG002"], col_names=["REF001", "REF002"],
                                     data=np.array([[0.5, -0.25], [0.125, 0.75]])),
                           rowcals=np.ones(2), colcals=np.ones(2))]
info9._update_redundant()
raw9 = mne.io.RawArray(rng.normal(scale=1e-12, size=(4, 200)), info9)
record("comp", save(raw9, "comp", buffer_size_sec=1.0))

# 7. long channel names (the extended channel information), a name with a colon
info10 = make_info(["eeg", "eeg"])
with info10._unlock():
    info10["chs"][0]["ch_name"] = "a_rather_long_channel_name_0001"
    info10["chs"][1]["ch_name"] = "EEG 002"
    info10._update_redundant()
    info10["bads"] = ["a_rather_long_channel_name_0001"]
raw10 = mne.io.RawArray(rng.normal(scale=1e-5, size=(2, 120)), info10)
record("longnames", save(raw10, "longnames", buffer_size_sec=1.0))

# refusals: epochs, evoked, a covariance, not a FIF file, a truncated file, a split file whose next part is missing, a MaxShield file
ep = mne.EpochsArray(rng.normal(size=(3, 2, 20)) * 1e-6, make_info(["eeg", "eeg"]), tmin=-0.1, verbose="ERROR")
ep.save(data_dir / "fif_refuse_epo.fif", overwrite=True, verbose="ERROR")
ev = mne.EvokedArray(rng.normal(size=(2, 20)) * 1e-6, make_info(["eeg", "eeg"]), tmin=-0.1, verbose="ERROR")
ev.save(data_dir / "fif_refuse_ave.fif", overwrite=True, verbose="ERROR")
cov = mne.make_ad_hoc_cov(make_info(["eeg", "eeg"]))
mne.write_cov(data_dir / "fif_refuse_cov.fif", cov, overwrite=True, verbose="ERROR")
(data_dir / "fif_refuse_notfif.fif").write_bytes(b"this is not a fif file, only text\n" * 4)
whole = (data_dir / "fif_short_raw.fif").read_bytes()
(data_dir / "fif_refuse_truncated_raw.fif").write_bytes(whole[: len(whole) * 3 // 4])
orph = mne.io.RawArray(rng.normal(scale=1e-5, size=(2, 1500)), make_info(["eeg", "eeg"]))
orph.save(data_dir / "fif_orphan_raw.fif", overwrite=True, split_size=8000, buffer_size_sec=0.5, verbose="ERROR")
for extra_part in data_dir.glob("fif_orphan_raw-*.fif"):
    extra_part.unlink()

(here / "fif_golden.json").write_text(json.dumps(golden, indent=1))
print("wrote", len(golden["cases"]), "cases;", sum(f.stat().st_size for f in data_dir.glob("fif_*")), "bytes of data")
