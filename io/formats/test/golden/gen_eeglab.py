#!/usr/bin/env python3
"""Synthetic EEGLAB datasets (.set, .fdt) and their goldens, made with MNE-Python (a reference for parity only: nxr-neuro reads the format natively).
The .set files are written here with scipy (MATLAB v5 files), then read back with MNE for the goldens.

Writes test/data/eeglab_*.set|fdt and test/golden/eeglab_golden.json plus one raw float32 `.f32` per file. Run:
  uv run --project <nxr> python test/golden/gen_eeglab.py
"""
import json
import pathlib
import warnings
import numpy as np
import scipy.io as sio
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
data_dir.mkdir(exist_ok=True)
rng = np.random.default_rng(20261117)
golden = {"mne": mne.__version__, "cases": {}}

LABELS = ["Fp1", "Fp2", "Cz", "Pz"]
XYZ = [(80.1, 30.2, 5.5), (80.1, -30.2, 5.5), (0.0, 0.0, 95.0), (-60.0, 0.0, 70.0)]
EVENTS = [("stim_a", 101, 0), ("stim_b", 301.5 + 0.5, 0), ("resp", 451, 5), ("boundary", 600, 50)]   # type, latency (1-based sample), duration (samples)
N = 800
SRATE = 250.0


def struct_array(rows, fields):
    arr = np.empty((1, len(rows)), dtype=[(f, "O") for f in fields])
    for j, r in enumerate(rows):
        for f in fields:
            arr[0, j][f] = r[f]
    return arr


def eeg_struct(data_field, nbchan=4, pnts=N, trials=1, extra=None):
    chanlocs = struct_array([dict(labels=LABELS[i], type="EEG", X=XYZ[i][0], Y=XYZ[i][1], Z=XYZ[i][2]) for i in range(nbchan)], ["labels", "type", "X", "Y", "Z"])
    events = struct_array([dict(type=t, latency=float(l), duration=float(d), urevent=float(i + 1)) for i, (t, l, d) in enumerate(EVENTS)], ["type", "latency", "duration", "urevent"])
    eeg = dict(setname="synthetic", filename="x.set", filepath="", subject="", group="", condition="", session=np.array([[]]), comments="made up",
               nbchan=float(nbchan), trials=float(trials), pnts=float(pnts), srate=SRATE, xmin=0.0, xmax=(pnts - 1) / SRATE, times=np.array([[]]),
               data=data_field, icaact=np.array([[]]), icawinv=np.eye(nbchan), icasphere=np.eye(nbchan), icaweights=np.eye(nbchan), icachansind=np.arange(1, nbchan + 1, dtype=float),
               chanlocs=chanlocs, event=events, epoch=np.array([[]]), ref="common")
    if extra:
        eeg.update(extra)
    return eeg


def record(name, stem):
    raw = mne.io.read_raw_eeglab(data_dir / f"{stem}.set", preload=True, verbose="ERROR")
    d = raw.get_data().astype("<f4")
    (here / f"eeglab_{name}.f32").write_bytes(d.tobytes())
    ann = raw.annotations
    golden["cases"][name] = {
        "file": f"{stem}.set", "mne_channels": list(raw.ch_names), "sfreq": raw.info["sfreq"], "n_times": int(raw.n_times), "data": f"eeglab_{name}.f32",
        "annotations": [{"onset": float(o), "duration": float(du), "label": str(l)} for o, du, l in zip(ann.onset, ann.duration, ann.description)],
    }


sig = rng.normal(scale=15.0, size=(4, N)).astype(np.float32)   # microvolts

# 1. data in a .fdt (float32, channels x samples, sample after sample = Fortran order)
(data_dir / "eeglab_fdt.fdt").write_bytes(np.asfortranarray(sig).T.astype("<f4").tobytes())   # sample-major: channel fastest
sio.savemat(data_dir / "eeglab_fdt.set", {"EEG": eeg_struct("eeglab_fdt.fdt")}, do_compression=False)
record("fdt", "eeglab_fdt")
# 2. data inside the .set (uncompressed v5), single
sio.savemat(data_dir / "eeglab_inline.set", {"EEG": eeg_struct(sig)}, do_compression=False)
record("inline", "eeglab_inline")
# 3. data inside the .set, double
sio.savemat(data_dir / "eeglab_inline_double.set", {"EEG": eeg_struct(sig.astype(np.float64))}, do_compression=False)
record("inline_double", "eeglab_inline_double")
# 4. the fields as top-level variables (older EEGLAB), data in the file
top = eeg_struct(sig)
sio.savemat(data_dir / "eeglab_toplevel.set", top, do_compression=False)
# MNE does not read this layout; the data and header are the same as case 2
golden["cases"]["toplevel"] = dict(golden["cases"]["inline"], file="eeglab_toplevel.set")

# refusals
sio.savemat(data_dir / "eeglab_compressed.set", {"EEG": eeg_struct(sig)}, do_compression=True)
ep = rng.normal(size=(4, 100, 3)).astype(np.float32)
sio.savemat(data_dir / "eeglab_epoched.set", {"EEG": eeg_struct(ep, pnts=100, trials=3)}, do_compression=False)
(data_dir / "eeglab_fdt_short.fdt").write_bytes((data_dir / "eeglab_fdt.fdt").read_bytes()[:-8])
sio.savemat(data_dir / "eeglab_fdt_short.set", {"EEG": eeg_struct("eeglab_fdt_short.fdt")}, do_compression=False)
sio.savemat(data_dir / "eeglab_nofdt.set", {"EEG": eeg_struct("eeglab_missing.fdt")}, do_compression=False)
(data_dir / "eeglab_not_mat.set").write_bytes(b"this is not a MATLAB file" * 20)

(here / "eeglab_golden.json").write_text(json.dumps(golden, indent=1))
print("wrote", len(golden["cases"]), "cases")
