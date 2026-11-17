#!/usr/bin/env python3
"""A synthetic BIDS dataset and its goldens, made with MNE-BIDS (a reference for parity only: nxr-neuro parses BIDS natively).
MNE-BIDS writes the EEG recordings, channels, events, electrodes, coordinate systems, participants and scans files from generated signals; the
metadata files that test the inheritance principle and the odd files (a split recording, a non-BIDS name) are added by hand below. The goldens are
what MNE-BIDS reads back (entities, nearest sidecars, participants), plus the merged sidecars computed here from the specification's rule (a deeper
file's keys replace a shallower file's). No real data, no real subject: every identifying value is invented.

Writes test/data/bids/ (the tree) and test/golden/bids_golden.json. Run:
  uv run --project <nxr> --with mne-bids --with pybv --with edfio python test/golden/gen_bids.py
"""
import datetime
import json
import pathlib
import shutil
import tempfile
import warnings

import numpy as np
import mne
import mne_bids
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description, get_entities_from_fname, find_matching_paths

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
here = pathlib.Path(__file__).resolve().parent
root = here.parent / "data" / "bids"
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
tmp = pathlib.Path(tempfile.mkdtemp())
rng = np.random.default_rng(20261119)
NAMES = ["Fz", "Cz", "Pz", "Oz", "C3", "C4"]


def make_raw(tag, sub_info, events=True):
    data = rng.normal(size=(len(NAMES), 300)) * 1e-5
    info = mne.create_info(NAMES, 100.0, "eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020")
    raw.set_meas_date(datetime.datetime(2026, 3, 4, 5, 6, 7, tzinfo=datetime.timezone.utc))
    with raw.info._unlock():
        raw.info["line_freq"] = 50.0
        raw.info["subject_info"] = sub_info
    if events:
        raw.set_annotations(mne.Annotations([0.5, 1.5], [0.1, 0.2], ["stim", "resp"]))
    f = tmp / f"{tag}_raw.fif"
    raw.save(f, overwrite=True)
    return mne.io.read_raw_fif(f, preload=False)


def write(sub, ses, task, run, fmt="EDF", **kw):
    raw = make_raw(f"{sub}{ses}{task}{run}", dict(his_id=f"SYN{sub}", sex=1 if sub == "01" else 2, hand=1, birthday=datetime.date(1990 + int(sub), 1, 1)))
    bp = BIDSPath(subject=sub, session=ses, task=task, run=run, datatype="eeg", root=root)
    write_raw_bids(raw, bp, format=fmt, allow_preload=True, overwrite=True, verbose=False, **kw)
    return bp


write("01", "01", "rest", "01")
write("01", "01", "rest", "02")
write("01", "01", "oddball", None, fmt="BrainVision")
write("01", "02", "rest", None)
write("02", None, "rest", None, fmt="BrainVision")
make_dataset_description(path=root, name="Synthetic EEG", dataset_type="raw", data_license="CC0", authors=["A. Invented", "B. Imagined"],
                         how_to_acknowledge="Cite nothing: invented", funding=["Invented Fund 1"], ethics_approvals=["none"],
                         references_and_links=["https://example.invalid/ds"], doi="doi:10.0000/invented", overwrite=True)

# ---- by hand: inheritance, odd files, a derivative ----
def put(rel, content):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content if isinstance(content, str) else json.dumps(content, indent=4) + "\n")


put("task-rest_eeg.json", {"Manufacturer": "Invented Systems", "CustomKey": 1, "EEGReference": "root-ref", "Nested": {"a": 1, "b": 2}})
put("sub-01/sub-01_task-rest_eeg.json", {"CustomKey": 2, "SubjectLevel": True, "Nested": {"a": 9}})
put("sub-01/ses-01/sub-01_ses-01_task-rest_eeg.json", {"SessionLevel": "s1", "EEGReference": "session-ref"})
put("task-rest_events.json", {"onset": {"Description": "onset of the event"}})
put("README", "synthetic\n")
put("CHANGES", "1.0 invented\n")
put("sub-02/eeg/notes_misc.txt", "not a BIDS name\n")
put("sub-02/eeg/sub-02_task-rest_headshape.pos", "3\n0 0 0\n")
put("derivatives/pipeline-a/dataset_description.json", {"Name": "pipeline A", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
                                                       "GeneratedBy": [{"Name": "invented-pipeline", "Version": "0.1"}]})
# ---- goldens ----
def rel(p):
    return str(pathlib.Path(p).relative_to(root)).replace("\\", "/")


def table(p):
    lines = pathlib.Path(p).read_text().rstrip("\n").split("\n")
    return {"path": rel(p), "columns": lines[0].split("\t"), "rows": [l.split("\t") for l in lines[1:]]}


def applicable(target, suffix, ext, ignore=()):
    """the specification's rule, written independently of the C++: entities of the candidate are a subset of the target's"""
    t = get_entities_from_fname(pathlib.Path(target).name, on_error="ignore")
    t = {k: v for k, v in t.items() if v is not None}
    out = []
    parts = pathlib.Path(target).relative_to(root).parts[:-1]
    for depth in range(len(parts) + 1):
        d = root.joinpath(*parts[:depth])
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            name = f.name
            stem, _, e = name.partition(".")
            if "." + e != ext or not stem.endswith("_" + suffix) and stem != suffix:
                continue
            ents = get_entities_from_fname(name, on_error="ignore")
            ents = {k: v for k, v in ents.items() if v is not None and k not in ignore}
            if all(t.get(k) == v for k, v in ents.items()):
                out.append((depth, len(ents), rel(f)))
    return [p for _, _, p in sorted(out, key=lambda x: (x[0], x[1]))]


golden = {"mne_bids": mne_bids.__version__, "recordings": [], "names": [], "build": []}
for p in sorted(root.glob("sub-*/**/*_eeg.edf")) + sorted(root.glob("sub-*/**/*_eeg.vhdr")):
    ents = {k: v for k, v in get_entities_from_fname(p.name, on_error="ignore").items() if v is not None}
    rec = {"data_files": [rel(p)], "entities": ents, "datatype": "eeg"}
    rec["companions"] = [rel(q) for q in sorted(p.parent.glob(p.stem + ".*")) if q.suffix in (".vmrk", ".eeg")]
    target = p
    sidecars = applicable(target, "eeg", ".json")
    merged = {}
    for s in sidecars:
        merged.update(json.loads((root / s).read_text()))
    rec["sidecar_files"] = sidecars
    rec["sidecar"] = merged
    for what, suffix in (("channels", "channels"), ("events", "events")):
        a = applicable(target, suffix, ".tsv")
        rec[what] = table(root / a[-1]) if a else None
    rec["electrodes"] = [table(root / x) for x in applicable(target, "electrodes", ".tsv", ignore=("space",))]
    rec["coordsystems"] = applicable(target, "coordsystem", ".json", ignore=("space",))
    # the nearest of each, as MNE-BIDS finds it
    bp = BIDSPath(root=root, **{k: v for k, v in ents.items() if k in ("subject", "session", "task", "run", "acquisition", "processing", "space", "recording", "description")}, datatype="eeg")
    found = bp.find_matching_sidecar(suffix="channels", extension=".tsv", on_error="ignore")
    rec["mne_bids_channels"] = None if found is None else rel(found)   # the nearest channels file as MNE-BIDS finds it
    golden["recordings"].append(rec)

golden["description"] = json.loads((root / "dataset_description.json").read_text())
golden["participants"] = table(root / "participants.tsv")
golden["participants_json"] = json.loads((root / "participants.json").read_text())
golden["scans"] = {rel(p): table(p) for p in sorted(root.glob("sub-*/**/*_scans.tsv"))}
golden["subjects"] = {s.name[4:]: [x.name[4:] for x in sorted(s.glob("ses-*"))] for s in sorted(root.glob("sub-*"))}
golden["top_level_files"] = sorted(p.name for p in root.iterdir() if p.is_file() and p.name not in ("dataset_description.json", "participants.tsv", "participants.json"))
golden["derivatives"] = ["derivatives/" + p.name for p in sorted((root / "derivatives").iterdir())]

# file names: valid and invalid, with what MNE-BIDS reads from the valid ones it knows
cases = [
    "sub-01_ses-a_task-rest_run-02_eeg.edf", "sub-01_task-rest_eeg.fif", "sub-01_task-rest_split-01_meg.fif", "sub-01_space-CapTrak_electrodes.tsv",
    "sub-01_coordsystem.json", "task-rest_eeg.json", "participants.tsv", "sub-01_acq-hi_proc-clean_desc-x_eeg.set", "sub-01_T1w.nii.gz",
    "sub-01_task-rest_run-1_events.tsv", "sub-01_task-rest_eeg", "sub-01_acq-a1_echo-2_flip-3_bold.nii.gz",
    # refused
    "sub-_task-rest_eeg.edf", "sub-01_task_eeg.edf", "sub-01_task-re_st_eeg.edf", "sub-01_task-rest_task-rest_eeg.edf", "sub-01_ta sk-rest_eeg.edf",
    ".bidsignore", "notes_misc.txt", "sub-01_task-rest_ee-g.edf", "dataset_description.json",
]
for c in cases:
    e = {"name": c}
    try:
        ents = get_entities_from_fname(c, on_error="raise")
        e["mne_bids"] = {k: v for k, v in ents.items() if v is not None}
    except Exception:
        e["mne_bids"] = None
    golden["names"].append(e)

# names built in the BIDS entity order (MNE-BIDS's own order differs for a few entities, so only entities both order alike are compared)
for kw in (dict(subject="01", session="a", task="rest", run="02", suffix="eeg", extension=".edf"),
           dict(subject="01", task="rest", acquisition="hi", processing="clean", description="x", suffix="eeg", extension=".set"),
           dict(subject="01", space="CapTrak", datatype="eeg", suffix="electrodes", extension=".tsv"),
           dict(subject="01", session="1", task="rest", split="02", suffix="meg", extension=".fif")):
    golden["build"].append({"entities": {k: v for k, v in kw.items() if k != "datatype"}, "name": BIDSPath(**kw).basename})

with open(here / "bids_golden.json", "w") as f:
    json.dump(golden, f, indent=1)
size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
print("tree", size, "bytes in", sum(1 for p in root.rglob("*") if p.is_file()), "files")
