#!/usr/bin/env python3
"""Synthetic EDF, EDF+ (continuous and discontinuous), BDF and BDF+ files and their goldens, made with MNE-Python (a reference for parity only:
nxr-neuro reads these formats natively). The files are written here with numpy (the format is simple), then read back with MNE for the goldens.

Writes test/data/edf_*.edf|bdf and test/golden/edf_golden.json plus one raw float32 `.f32` per file (channel after channel, samples in order).
Run:  uv run --project <nxr> python test/golden/gen_edf.py
"""
import json
import pathlib
import warnings
import numpy as np
import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
data_dir.mkdir(exist_ok=True)
rng = np.random.default_rng(20261117)


def pad(s, n):
    b = s.encode("latin-1") if isinstance(s, str) else s
    assert len(b) <= n, (s, n)
    return b.ljust(n, b" ")


def fmt_num(v, n):
    s = repr(float(v)) if float(v) != int(v) else str(int(v))
    s = s[:n]
    return pad(s, n)


def tal(onset, duration, labels):
    s = ("%+.7f" % onset).rstrip("0").rstrip(".")
    if duration is not None:
        s += "\x15" + ("%.7f" % duration).rstrip("0").rstrip(".")
    s += "\x14" + "".join(l + "\x14" for l in labels) + "\x00"
    return s.encode("utf-8")


def write_edf(path, signals, rec_dur, n_rec, bdf=False, plus=None, rec_starts=None, annotations=None, patient="X X X X", recording="Startdate X X X X"):
    """signals: list of dict(label, dim, pmin, pmax, dmin, dmax, nsamp, digital (n_rec*nsamp ints), transducer, prefilter).
    plus: None | 'C' | 'D'.  rec_starts: onset (s) of each record (EDF+); annotations: [(onset, duration|None, label)] placed in the record they fall in."""
    bps = 3 if bdf else 2
    sigs = list(signals)
    ann_samples = 0
    if plus:
        # annotations signal: sized for the time-keeping TAL plus the record's annotations
        recs_ann = [[] for _ in range(n_rec)]
        starts = rec_starts if rec_starts is not None else [k * rec_dur for k in range(n_rec)]
        for (on, du, lab) in (annotations or []):
            k = max(i for i in range(n_rec) if starts[i] <= on + 1e-9)
            recs_ann[k].append((on, du, lab))
        blobs = []
        for k in range(n_rec):
            b = tal(starts[k], None, [""]) + b"".join(tal(on, du, [lab]) for (on, du, lab) in recs_ann[k])
            blobs.append(b)
        ann_bytes = max(len(b) for b in blobs)
        ann_bytes += ann_bytes % bps and (bps - ann_bytes % bps)
        ann_samples = ann_bytes // bps
        sigs.append(dict(label="BDF Annotations" if bdf else "EDF Annotations", dim="", pmin=-1, pmax=1, dmin=-(2**(8*bps-1)), dmax=2**(8*bps-1)-1,
                         nsamp=ann_samples, transducer="", prefilter="", digital=None))
    ns = len(sigs)
    hdr = bytearray()
    hdr += (b"\xffBIOSEMI" if bdf else pad("0", 8))
    hdr += pad(patient, 80) + pad(recording, 80)
    hdr += pad("01.02.03", 8) + pad("04.05.06", 8)
    hdr += pad(str(256 * (ns + 1)), 8)
    hdr += pad(("BDF+" if bdf else "EDF+") + plus if plus else "", 44)
    hdr += pad(str(n_rec), 8) + fmt_num(rec_dur, 8) + pad(str(ns), 4)
    for key, n in [("label", 16), ("transducer", 80), ("dim", 8)]:
        for s in sigs:
            hdr += pad(s.get(key, ""), n)
    for key in ["pmin", "pmax", "dmin", "dmax"]:
        for s in sigs:
            hdr += fmt_num(s[key], 8)
    for s in sigs:
        hdr += pad(s.get("prefilter", ""), 80)
    for s in sigs:
        hdr += pad(str(s["nsamp"]), 8)
    for s in sigs:
        hdr += pad("", 32)
    assert len(hdr) == 256 * (ns + 1)
    body = bytearray()
    for k in range(n_rec):
        for s in sigs:
            if s["digital"] is None:
                b = blobs[k].ljust(ann_samples * bps, b"\x00")
                body += b
            else:
                v = s["digital"][k * s["nsamp"]:(k + 1) * s["nsamp"]].astype("<i4")
                if bdf:
                    raw = v.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
                else:
                    raw = v.astype("<i2").tobytes()
                body += raw
    pathlib.Path(path).write_bytes(bytes(hdr) + bytes(body))


def make_signal(label, dim, pmin, pmax, nsamp_rec, n_rec, bdf=False, **kw):
    dmin, dmax = (-8388608, 8388607) if bdf else (-32768, 32767)
    dig = rng.integers(dmin // 4, dmax // 4, size=nsamp_rec * n_rec)
    return dict(label=label, dim=dim, pmin=pmin, pmax=pmax, dmin=dmin, dmax=dmax, nsamp=nsamp_rec, digital=dig, **kw)


golden = {"mne": mne.__version__, "cases": {}}


def record(name, fname, extra=None, mne_ok=True):
    path = data_dir / fname
    case = {"file": fname}
    if mne_ok:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR") if fname.endswith("edf") else mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
        data = raw.get_data().astype("<f4")
        (here / f"edf_{name}.f32").write_bytes(data.tobytes())
        ann = raw.annotations
        case.update({
            "mne_channels": list(raw.ch_names), "sfreq": raw.info["sfreq"], "n_times": int(raw.n_times), "data": f"edf_{name}.f32",
            "annotations": [{"onset": float(o), "duration": float(d), "label": str(l)} for o, d, l in zip(ann.onset, ann.duration, ann.description)],
        })
    if extra:
        case.update(extra)
    golden["cases"][name] = case


# 1. plain EDF: 3 channels at 100 Hz (1 s records), different units and ranges
n_rec = 6
sigs = [make_signal("Fp1", "uV", -3276.8, 3276.7, 100, n_rec, transducer="AgAgCl electrode", prefilter="HP:0.1Hz LP:75Hz"),
        make_signal("Cz", "mV", -2.0, 2.0, 100, n_rec),
        make_signal("EOG", "uV", -500, 500, 100, n_rec)]
write_edf(data_dir / "edf_plain.edf", sigs, 1.0, n_rec)
record("plain", "edf_plain.edf", {"format": "edf", "n_gaps": 0})

# 2. EDF+C with annotations (instants and durations), 0.5 s records
n_rec = 8
sigs = [make_signal("EEG A1", "uV", -1000, 1000, 50, n_rec), make_signal("EEG A2", "uV", -1000, 1000, 50, n_rec)]
anns = [(0.0, None, "start"), (0.75, 0.5, "stimulus"), (2.25, None, "response"), (3.1, 0.4, "blink")]
write_edf(data_dir / "edf_plus_c.edf", sigs, 0.5, n_rec, plus="C", annotations=anns)
record("plus_c", "edf_plus_c.edf", {"format": "edf+", "n_gaps": 0})

# 3. EDF+D: records 0-2 contiguous, then a 2 s jump (record 3 starts at 5.0 instead of 3.0), then contiguous; 1 s records at 20 Hz
n_rec = 6
sigs = [make_signal("Ch1", "uV", -100, 100, 20, n_rec), make_signal("Ch2", "uV", -100, 100, 20, n_rec)]
starts = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
write_edf(data_dir / "edf_plus_d.edf", sigs, 1.0, n_rec, plus="D", rec_starts=starts, annotations=[(5.5, 0.25, "after gap")])
record("plus_d", "edf_plus_d.edf", {"format": "edf+", "gaps": [{"first": 60, "count": 40}], "n_samples": 160}, mne_ok=False)
# the sample positions of the data, from the construction: record k occupies [start_k * 20, start_k * 20 + 20)
dig = np.stack([s["digital"] for s in sigs])
cal = [(s["pmax"] - s["pmin"]) / (s["dmax"] - s["dmin"]) for s in sigs]
full = np.zeros((2, 160), dtype=np.float32)
for k in range(n_rec):
    a = int(round(starts[k] * 20))
    for c, s in enumerate(sigs):
        d = s["digital"][k * 20:(k + 1) * 20]
        full[c, a:a + 20] = ((d - s["dmin"]) * cal[c] + s["pmin"]) * 1e-6
(here / "edf_plus_d.f32").write_bytes(full.astype("<f4").tobytes())
golden["cases"]["plus_d"].update({"data": "edf_plus_d.f32", "sfreq": 20.0, "mne_channels": ["Ch1", "Ch2"],
                                  "annotations": [{"onset": 5.5, "duration": 0.25, "label": "after gap"}]})

# 4. BDF with a Status channel, 4 channels at 250 Hz
n_rec = 4
sigs = [make_signal(f"A{i}", "uV", -262144, 262143, 250, n_rec, bdf=True) for i in range(1, 4)]
status = make_signal("Status", "Boolean", -8388608, 8388607, 250, n_rec, bdf=True)
status["digital"] = (rng.integers(0, 3, size=250 * n_rec) * 256).astype(np.int64)
sigs.append(status)
write_edf(data_dir / "bdf_plain.bdf", sigs, 1.0, n_rec, bdf=True)
record("bdf_plain", "bdf_plain.bdf", {"format": "bdf", "n_gaps": 0})

# 5. BDF+ with annotations
n_rec = 4
sigs = [make_signal(f"B{i}", "uV", -262144, 262143, 128, n_rec, bdf=True) for i in range(1, 3)]
write_edf(data_dir / "bdf_plus.bdf", sigs, 1.0, n_rec, bdf=True, plus="C", annotations=[(1.5, 0.5, "event one"), (3.0, None, "event two")])
record("bdf_plus", "bdf_plus.bdf", {"format": "bdf+", "n_gaps": 0})

# 6. two sampling rates (refused unless a rate is named): 100 Hz x2 and 25 Hz x1
n_rec = 4
sigs = [make_signal("Fast1", "uV", -100, 100, 100, n_rec), make_signal("Slow", "uV", -100, 100, 25, n_rec), make_signal("Fast2", "uV", -100, 100, 100, n_rec)]
write_edf(data_dir / "edf_mixed_rates.edf", sigs, 1.0, n_rec)
cal = [(s["pmax"] - s["pmin"]) / (s["dmax"] - s["dmin"]) for s in sigs]
def phys(i):
    s = sigs[i]
    return (((s["digital"] - s["dmin"]) * cal[i] + s["pmin"]) * 1e-6).astype("<f4")
(here / "edf_mixed_fast.f32").write_bytes(np.stack([phys(0), phys(2)]).tobytes())
(here / "edf_mixed_slow.f32").write_bytes(phys(1).tobytes())
golden["cases"]["mixed_rates"] = {"file": "edf_mixed_rates.edf", "fast": "edf_mixed_fast.f32", "slow": "edf_mixed_slow.f32"}

# 7. refusals: truncated data, a bad header
raw = (data_dir / "edf_plain.edf").read_bytes()
(data_dir / "edf_truncated.edf").write_bytes(raw[:-300])
(data_dir / "edf_bad_version.edf").write_bytes(b"1" + raw[1:])

(here / "edf_golden.json").write_text(json.dumps(golden, indent=1))
print("wrote", len(golden["cases"]), "cases")
