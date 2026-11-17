#!/usr/bin/env python3
"""Synthetic CTF data sets and their goldens. MNE-Python cannot write CTF, so this script writes minimal valid data sets from the format description
(the .res4 layout, big-endian int32 .meg4 trials, the text files); MNE-Python reading them (`mne.io.read_raw_ctf`) is the check that they are valid and
the source of the goldens. No real data, no real subject: every value is invented.

Writes test/data/ctf/*.ds and test/golden/ctf_golden.json, and a raw .f32 per readable data set (channel after channel, as MNE reads it). Run:
  uv run --project <nxr> python test/golden/gen_ctf.py
"""
import json
import pathlib
import shutil
import struct
import warnings

import numpy as np
import mne
from mne.io.ctf.res4 import _read_res4
from mne.io.ctf.hc import _read_hc
from mne.io.ctf.eeg import _read_eeg, _read_pos
from mne.io.ctf.trans import _make_ctf_coord_trans_set

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
here = pathlib.Path(__file__).resolve().parent
root = here.parent / "data" / "ctf"
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
rng = np.random.default_rng(20261120)
golden = {"mne": mne.__version__, "cases": {}, "refused": {}}
FUNNY_POS = 1844


class W:
    def __init__(self):
        self.b = bytearray()

    def s(self, text, n):
        d = text.encode("utf-8")[:n]
        self.b += d + b"\0" * (n - len(d))

    def i2(self, v): self.b += struct.pack(">h", v)
    def i4(self, v): self.b += struct.pack(">i", v)
    def f8(self, v): self.b += struct.pack(">d", v)
    def u1(self, v): self.b += struct.pack(">B", v)

    def align(self, k):
        while len(self.b) % k:
            self.b += b"\0"

    def seek_pad(self, pos):
        assert len(self.b) <= pos, (len(self.b), pos)
        self.b += b"\0" * (pos - len(self.b))


def coil(w, pos_cm, norm, turns=1, area_cm2=1.0):
    for x in pos_cm: w.f8(x)
    w.f8(0.0)
    for x in norm: w.f8(x)
    w.f8(0.0)
    w.i2(turns); w.i4(0); w.i2(0)
    w.f8(area_cm2)


NOCOIL = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def make_res4(chs, comps, nsamp, sfreq, *, pre_trig=0, nave=1, no_trials=1, date="05/03/2026", time="14:30:15", run_desc="synthetic run", filters=((100.0, 0, 1), (0.5, 0, 2))):
    w = W()
    w.s("MEG41RS", 8)
    w.s("synthetic-app", 256); w.s("synthetic-origin", 256); w.s("synthetic description", 256)
    w.i2(nave)
    w.s(time, 255); w.s(date, 255)
    w.i4(nsamp); w.i2(len(chs)); w.align(8)
    w.f8(sfreq); w.f8(nsamp / sfreq)
    w.i2(no_trials); w.align(4)
    w.i4(pre_trig)
    w.i2(no_trials); w.i2(0); w.align(4)
    w.i4(1)
    w.u1(65); [w.u1(66 + k) for k in range(8)]; w.u1(67)
    w.i2(3); w.align(4)
    w.i4(0)
    w.i2(0); w.align(4)
    w.i4(0); w.align(4)
    w.i4(1)
    w.i4(0)
    w.s("run-0001", 32); w.s("synthetic run title", 256); w.s("instrument", 32); w.s("descriptor", 32); w.s("SYNTH-001", 32); w.s("OperatorX", 32)
    w.s("sensor.file", 60); w.align(4)
    w.i4(len(run_desc))
    w.seek_pad(FUNNY_POS)
    w.b += run_desc.encode()
    w.i2(len(filters))
    for freq, klass, typ in filters:
        w.f8(freq); w.i4(klass); w.i4(typ); w.i2(2); w.f8(1.5); w.f8(2.5)
    for c in chs:
        w.s(c["name"], 32)
    for c in chs:
        w.i2(c["type"]); w.i2(1); w.i4(0)
        w.f8(c["pg"]); w.f8(c["qg"]); w.f8(c["iog"]); w.f8(c["ioo"])
        w.i2(c.get("ncoil", 1)); w.i2(c.get("order", 0)); w.i4(0)
        coils = c.get("coils", [NOCOIL])
        for k in range(8):
            p, n = coils[k] if k < len(coils) else NOCOIL
            coil(w, p, n, area_cm2=c.get("area", 2.0))
        for k in range(8):
            coil(w, *NOCOIL)
    w.i2(len(comps))
    for row in comps:
        w.s(row["sensor"], 32); w.i4(row["kind"]); w.i4(0); w.i2(len(row["refs"]))
        for k in range(50):
            w.s(row["refs"][k] if k < len(row["refs"]) else "", 31)
        for k in range(50):
            w.f8(row["coeffs"][k] if k < len(row["coeffs"]) else 0.0)
    return bytes(w.b)


def hc_text(dev, head, spare=True):
    names = [("nasion", "nasion"), ("left ear", "left"), ("right ear", "right")] + ([("spare", "spare")] if spare else [])
    lines = []
    for i, (title, _) in enumerate(names):
        for frame, pts in (("dewar", dev), ("head", head)):
            for state in ("standard", "measured"):
                lines.append(f"{state} {title} coil position relative to {frame} (cm):")
                for ax, v in zip("xyz", pts[i]):
                    lines.append(f"\t{ax} = {v:.4f}")
    return "\n".join(lines) + "\n"


def write_ds(name, chs, comps, trials, nsamp, sfreq, *, extra_meg4=None, hc=True, files=None, res4_kw=None, clock_zero_at=None, meg4_raw=None):
    d = root / f"{name}.ds"
    d.mkdir()
    (d / f"{name}.res4").write_bytes(make_res4(chs, comps, nsamp, sfreq, **(res4_kw or {})))

    def meg4(trials_data):
        return b"MEG41CP\0" + b"".join(t.astype(">i4").tobytes() for t in trials_data)

    (d / f"{name}.meg4").write_bytes(meg4_raw if meg4_raw is not None else meg4(trials))
    if extra_meg4 is not None:
        (d / f"{name}.1_meg4").write_bytes(meg4(extra_meg4))
    if hc:
        dev = np.array([[0.0, 8.0, -3.0], [-7.5, 0.0, -2.5], [7.5, 0.5, -2.0], [0.5, -1.0, 4.0]])
        th = np.deg2rad(7.0)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        head = (dev - np.array([0.0, 0.2, 0.0])) @ R.T + np.array([0.3, -0.1, 0.4])
        # head frame: nasion ahead (x), ears on y: build from the landmark geometry by hand
        nas, lpa, rpa = np.array([9.0, 0.0, 0.0]), np.array([0.0, 7.0, 0.0]), np.array([0.0, -7.0, 0.0])
        head_pts = np.array([nas, lpa, rpa, [2.0, 1.0, 5.0]])
        # device points: any rigid transform of the head points
        th = np.deg2rad(11.0)
        R = np.array([[np.cos(th), -np.sin(th), 0.1], [np.sin(th), np.cos(th), 0.0], [-0.1, 0.0, 1.0]])
        q, _ = np.linalg.qr(R)
        dev = head_pts @ q.T + np.array([0.5, -0.3, -4.0])
        (d / f"{name}.hc").write_text(hc_text(dev, head_pts))
    for fn, content in (files or {}).items():
        (d / fn).write_text(content)
    return d


def chans(eeg_pos=True):
    def meg(name, pos, norm, order=3, pg=2.0e6, qg=3.0e12):
        return dict(name=name, type=5, pg=pg, qg=qg, iog=1.0, ioo=0.0, order=order, coils=[(pos, norm), (pos, norm)])
    chs = [
        meg("MLC11-4408", [3.0, 4.0, 5.0], [0.0, 0.0, 1.0]),
        meg("MLC12-4408", [-3.0, 4.5, 5.5], [0.0, 0.6, 0.8]),
        meg("MRC11-4408", [6.0, -2.0, 4.0], [0.6, 0.0, 0.8], pg=-1.5e6),
        dict(name="MLZ99-4408", type=5, pg=2.0e6, qg=3.0e12, iog=1.0, ioo=0.0, order=3, coils=[NOCOIL]),
        dict(name="BG1-4408", type=0, pg=1.0e6, qg=4.0e12, iog=1.0, ioo=0.0, coils=[([1.0, 1.0, 9.0], [0.0, 0.0, 1.0])]),
        dict(name="G11-4408", type=1, pg=1.0e6, qg=4.0e12, iog=1.0, ioo=0.0, ncoil=2, coils=[([1.0, 2.0, 9.0], [0.0, 0.0, 1.0]), ([1.0, 2.0, 10.0], [0.0, 0.0, -1.0])]),
        dict(name="G12-4408", type=1, pg=1.0e6, qg=4.0e12, iog=1.0, ioo=0.0, ncoil=2, coils=[([2.0, 2.0, 9.0], [0.0, 0.0, 1.0]), ([3.0, 2.0, 9.0], [0.0, 0.0, -1.0])]),
        dict(name="EEG01", type=9, pg=1.0, qg=2.0e5, iog=1.0, ioo=0.0, coils=[([5.0, 2.0, 7.0], [0.0, 0.0, 1.0])] if eeg_pos else [NOCOIL]),
        dict(name="UPPT001", type=11, pg=1.0, qg=1.0, iog=1.0, ioo=0.0, coils=[NOCOIL]),
        dict(name="SCLK01-177", type=14, pg=1.0, qg=1.0, iog=1.0, ioo=0.0, coils=[NOCOIL]),
    ]
    return chs


def comps_for(chs):
    megs = [c["name"] for c in chs if c["type"] == 5 and np.abs(np.asarray(c["coils"][0][0])).sum() > 0]
    refs = ["BG1-4408", "G11-4408"]
    return [dict(sensor=m, kind=0x47334252, refs=refs, coeffs=[0.01 * (i + 1), -0.02 * (i + 1)]) for i, m in enumerate(megs)]


def trial_data(chs, nsamp, n_trials, t0=0, zero_clock_at=None):
    out = []
    for t in range(n_trials):
        a = rng.integers(-20000, 20000, size=(len(chs), nsamp)).astype(np.int32)
        a[-1] = np.arange(t0 + t * nsamp + 1, t0 + (t + 1) * nsamp + 1)   # the clock counts up from 1
        if zero_clock_at is not None:
            g = np.arange(t0 + t * nsamp, t0 + (t + 1) * nsamp)
            a[-1][g >= zero_clock_at] = 0
        out.append(a)
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, bytes):
        return x.decode("latin1")
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if hasattr(x, "items"):
        return {k: jsonable(v) for k, v in x.items()}
    return x


def golden_for(case, d, *, system_clock="truncate"):
    res4 = _read_res4(str(d))
    coils = _read_hc(str(d))
    eeg = _read_eeg(str(d))
    trans = _make_ctf_coord_trans_set(res4, coils) if coils is not None else None
    raw = mne.io.read_raw_ctf(d, system_clock=system_clock, preload=True, verbose=False)
    digs = _read_pos(str(d), trans) if trans is not None else []
    data = raw.get_data()
    (here / f"ctf_{case}.f32").write_bytes(data.astype("<f4").tobytes())
    info = raw.info
    g = {"res4": jsonable(res4), "coils": jsonable(coils), "eeg": jsonable(eeg), "pos": jsonable(digs),
         "trans": None if trans is None else {k: (None if v is None else np.asarray(v["trans"] if hasattr(v, "keys") and "trans" in v else v).tolist()) for k, v in trans.items()},
         "n_times": raw.n_times, "sfreq": info["sfreq"], "ch_names": raw.ch_names,
         "kinds": [int(c["kind"]) for c in info["chs"]], "coil_types": [int(c["coil_type"]) & 0xFFFF for c in info["chs"]],
         "grades": [int(c["coil_type"]) >> 16 for c in info["chs"]], "units": [int(c["unit"]) for c in info["chs"]], "cals": [float(c["cal"]) for c in info["chs"]],
         "bads": info["bads"], "meas_date": info["meas_date"].isoformat() if info["meas_date"] else None,
         "annotations": [[float(a["onset"]), float(a["duration"]), a["description"]] for a in raw.annotations],
         "highpass": info["highpass"], "lowpass": info["lowpass"], "comp_grade": [int(c["kind"]) for c in info["comps"]],
         "subject": info["subject_info"], "experimenter": info["experimenter"]}
    golden["cases"][case] = g


# ---- A: continuous, 3 trials in the first file and one in a continuation file, every side file ----
nsamp, sfreq = 120, 600.0
A = chans()
ds = write_ds("ctf_cont", A, comps_for(A), trial_data(A, nsamp, 3), nsamp, sfreq, extra_meg4=trial_data(A, nsamp, 1, t0=3 * nsamp),
              files={"BadChannels": "MLC12\n",
                     "bad.segments": "blink 0.10 0.15\nmuscle 0.60 0.70\n",
                     "MarkerFile.mrk": "PATH OF DATASET:\n/x/ctf_cont.ds\n\n\nNUMBER OF MARKERS:\n2\n\n\nCLASSGROUPID:\n3\nNAME:\nstimA\nCOMMENT:\n\nCOLOR:\nred\nEDITABLE:\nYes\nCLASSID:\n1\nNUMBER OF SAMPLES:\n2\nLIST OF SAMPLES:\nTRIAL NUMBER\t\tTIME FROM SYNC POINT (in seconds)\n                  +0\t\t\t\t+0.250\n                  +2\t\t\t\t+0.100\n\n\nCLASSGROUPID:\n3\nNAME:\nstimB\nCOMMENT:\n\nCOLOR:\nblue\nEDITABLE:\nYes\nCLASSID:\n2\nNUMBER OF SAMPLES:\n1\nLIST OF SAMPLES:\nTRIAL NUMBER\t\tTIME FROM SYNC POINT (in seconds)\n                  +3\t\t\t\t+0.050\n",
                     "ctf_cont.eeg": "1 EEG01 5.0 2.0 7.0\n2 Nasion 9.0 0.0 0.0\n",
                     "ctf_cont.pos": "1 0.0 0.0 10.0\n2 Fz 1.0 2.0 3.0\nEXTRA 2.0 3.0 4.0\n"})
golden_for("cont", ds)

# ---- B: the system clock drops to zero: the recording stops there (and is read whole with the clock ignored) ----
B = chans(eeg_pos=False)
dsb = write_ds("ctf_clock", B, comps_for(B), trial_data(B, nsamp, 3, zero_clock_at=200), nsamp, sfreq, hc=False)
golden_for("clock", dsb)
golden_for("clock_ignored", dsb, system_clock="ignore")

# ---- refusals ----
def refused(name, **kw):
    d = write_ds(name, chans(), comps_for(chans()), trial_data(chans(), nsamp, 1), nsamp, sfreq, hc=False, **kw)
    return d


refused("ctf_epoched", res4_kw=dict(pre_trig=30))
refused("ctf_averaged", res4_kw=dict(nave=12))
refused("ctf_badtrial", meg4_raw=b"MEG41CP\0" + b"\0" * 1000)
refused("ctf_notctf", meg4_raw=b"NOTCTF\0\0" + b"\0" * (nsamp * len(chans()) * 4))
shutil.rmtree(root / "ctf_nodata.ds", ignore_errors=True)
d = refused("ctf_nodata")
(d / "ctf_nodata.meg4").unlink()
mixed = chans()
mixed[1]["order"] = 2
refused_d = write_ds("ctf_badgrade", mixed, comps_for(mixed), trial_data(mixed, nsamp, 1), nsamp, sfreq, hc=False)
nohdr = root / "ctf_nores4.ds"
nohdr.mkdir()
(nohdr / "notes.txt").write_text("a data set folder with no .res4 file\n")   # git keeps no empty folder

def clean(x):
    """plain JSON: arrays as lists, bytes as text, NaN and infinity as null (JSON has neither)"""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, bytes):
        return x.decode("latin1")
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


with open(here / "ctf_golden.json", "w") as f:
    json.dump(clean(golden), f, separators=(",", ":"))
size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
print("ctf data", size, "bytes;", "golden", (here / "ctf_golden.json").stat().st_size)
