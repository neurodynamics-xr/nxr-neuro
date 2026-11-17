#!/usr/bin/env python3
"""Synthetic BrainVision files (.vhdr/.vmrk/.eeg) and their goldens, made with MNE-Python (a reference for parity only: nxr-neuro reads the
format natively). The files are written here with numpy, then read back with MNE for the goldens.

Writes test/data/bv_*.* and test/golden/brainvision_golden.json plus one raw float32 `.f32` per file. Run:
  uv run --project <nxr> python test/golden/gen_brainvision.py
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
golden = {"mne": mne.__version__, "cases": {}}


def write_bv(stem, names, res, units, fmt, orient, n, coords=None, markers=None, binary="INT_16", sfreq=250.0, codepage="UTF-8", ascii_data=False, drop=0):
    nch = len(names)
    dt = {"INT_16": "<i2", "INT_32": "<i4", "IEEE_FLOAT_32": "<f4"}[binary]
    if binary == "IEEE_FLOAT_32":
        data = rng.normal(scale=20.0, size=(nch, n)).astype(np.float32)
    else:
        lim = 2 ** (15 if binary == "INT_16" else 30) // 4
        data = rng.integers(-lim, lim, size=(nch, n)).astype(dt)
    raw = (data.T if orient == "MULTIPLEXED" else data).astype(dt).tobytes()
    if drop:
        raw = raw[:-drop]
    (data_dir / f"{stem}.eeg").write_bytes(raw)
    lines = ["Brain Vision Data Exchange Header File Version 1.0", "; synthetic", "", "[Common Infos]", f"Codepage={codepage}",
             f"DataFile={stem}.eeg", f"MarkerFile={stem}.vmrk", f"DataFormat={fmt}", f"DataOrientation={orient}", "DataType=TIME",
             f"NumberOfChannels={nch}", f"SamplingInterval={1e6 / sfreq:g}", "", "[Binary Infos]", f"BinaryFormat={binary}", "", "[Channel Infos]"]
    for i, (nm, r, u) in enumerate(zip(names, res, units), start=1):
        lines.append(f"Ch{i}={nm},,{r:g},{u}")
    if coords:
        lines += ["", "[Coordinates]"] + [f"Ch{i}={r:g},{t:g},{p:g}" for i, (r, t, p) in enumerate(coords, start=1)]
    lines += ["", "[Comment]", "A synthetic file.", ""]
    (data_dir / f"{stem}.vhdr").write_bytes("\r\n".join(lines).encode("utf-8" if codepage == "UTF-8" else "latin-1"))
    mk = ["Brain Vision Data Exchange Marker File, Version 1.0", "", "[Common Infos]", f"Codepage={codepage}", f"DataFile={stem}.eeg", "", "[Marker Infos]"]
    for i, m in enumerate(markers or [], start=1):
        mk.append(f"Mk{i}=" + ",".join(str(x) for x in m))
    (data_dir / f"{stem}.vmrk").write_bytes("\r\n".join(mk + [""]).encode("utf-8"))
    return data, np.array(res, dtype=np.float64)


def record(name, stem, extra=None):
    raw = mne.io.read_raw_brainvision(data_dir / f"{stem}.vhdr", preload=True, verbose="ERROR")
    d = raw.get_data().astype("<f4")
    (here / f"brainvision_{name}.f32").write_bytes(d.tobytes())
    ann = raw.annotations
    case = {"file": f"{stem}.vhdr", "mne_channels": list(raw.ch_names), "sfreq": raw.info["sfreq"], "n_times": int(raw.n_times), "data": f"brainvision_{name}.f32",
            "annotations": [{"onset": float(o), "duration": float(du), "label": str(l)} for o, du, l in zip(ann.onset, ann.duration, ann.description)]}
    montage = raw.get_montage()
    if montage is not None:
        pos = montage.get_positions()["ch_pos"]
        case["positions"] = {k: [float(x) for x in v] for k, v in pos.items()}
    if extra:
        case.update(extra)
    golden["cases"][name] = case


names = ["Fp1", "Fp2", "Cz", "Pz"]
markers = [("New Segment", "", 1, 1, 0, "20260102150405000000"), ("Stimulus", "S  1", 101, 1, 0), ("Stimulus", "S 12", 301, 1, 0),
           ("Response", "R  1", 451, 5, 0), ("Comment", "a, comma", 600, 1, 2)]
# 1. multiplexed INT_16 with a resolution and coordinates
write_bv("bv_mux_i16", names, [0.1, 0.1, 0.5, 1.0], ["µV"] * 4, "BINARY", "MULTIPLEXED", 800, binary="INT_16",
         coords=[(1, 108, -72), (1, 108, 72), (1, 0, 0), (1, 90, 180)], markers=[(m[0], m[1].replace(",", "\\1"), *m[2:]) for m in markers])
record("mux_i16", "bv_mux_i16", {"format": "brainvision"})
# 2. vectorized IEEE_FLOAT_32
write_bv("bv_vec_f32", names[:3], [1, 1, 1], ["µV"] * 3, "BINARY", "VECTORIZED", 500, binary="IEEE_FLOAT_32", markers=[("New Segment", "", 1, 1, 0, "20260102150405000000"), ("Stimulus", "S  5", 50, 1, 0)])
record("vec_f32", "bv_vec_f32")
# 3. multiplexed INT_32, millivolts, a Latin-1 header
write_bv("bv_mux_i32", names[:2], [0.001, 0.002], ["mV", "mV"], "BINARY", "MULTIPLEXED", 300, binary="INT_32", codepage="ANSI", markers=[("New Segment", "", 1, 1, 0, "20260102150405000000")])
record("mux_i32", "bv_mux_i32")
# refusals: ASCII data, an unknown binary format, truncated data, no header
write_bv("bv_ascii", names[:2], [1, 1], ["µV"] * 2, "ASCII", "MULTIPLEXED", 50, binary="INT_16")
write_bv("bv_truncated", names[:2], [1, 1], ["µV"] * 2, "BINARY", "MULTIPLEXED", 50, binary="INT_16", drop=3)
write_bv("bv_uint16", names[:2], [1, 1], ["µV"] * 2, "BINARY", "MULTIPLEXED", 50, binary="INT_16")
p = data_dir / "bv_uint16.vhdr"
p.write_bytes(p.read_bytes().replace(b"BinaryFormat=INT_16", b"BinaryFormat=UINT_16"))
(here / "brainvision_golden.json").write_text(json.dumps(golden, indent=1))
print("wrote", len(golden["cases"]), "cases")
