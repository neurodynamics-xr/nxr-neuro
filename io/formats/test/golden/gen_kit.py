#!/usr/bin/env python3
"""Synthetic KIT/Yokogawa files and their goldens. MNE-Python cannot write KIT files, so this script writes minimal valid ones from the format description
(the directory, system, channel, calibration, amplifier, acquisition, coregistration and digitization blocks, little-endian int16 samples); MNE-Python
reading them (`mne.io.read_raw_kit`, `mne.io.kit.kit.get_kit_info`) is the check that they are valid and the source of the goldens. No real data, no
real subject: every value is invented.

Writes test/data/kit/*.con and test/golden/kit_golden.json, and a raw .f32 per readable file (channel after channel). Run:
  uv run --project <nxr> python test/golden/gen_kit.py
"""
import json
import pathlib
import shutil
import struct
import warnings

import numpy as np
import mne
from mne.io.kit.kit import get_kit_info

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
here = pathlib.Path(__file__).resolve().parent
root = here.parent / "data" / "kit"
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
rng = np.random.default_rng(20261121)
golden = {"mne": mne.__version__, "cases": {}}
N_DIR = 30
CH_SIZE = 192


def pad(b, n):
    b = b[:n]
    return b + b"\0" * (n - len(b))


def channel_record(ch):
    t = ch["type"]
    b = struct.pack("<i", t)
    if t in (1, 2, 3, 4, 0x101, 0x102, 0x103, 0x104):
        b += struct.pack("<5d", *ch["loc"])
        if t in (1, 2):
            b += b"\0" * 16 + pad(ch.get("name", "").encode(), 6)
    elif t in (-1, -2, -3, -4):
        b += struct.pack("<i", ch.get("no", 1)) + b"\0" * 4
        width = {-1: 32, -2: 8, -3: 32, -4: 32}[t]
        b += pad(ch.get("name", "").encode(), width)
        if t in (-2, -3):
            b += b"\0" * (6 if t == -2 else 8) + struct.pack("<d", ch["exg_gain"])
    return pad(b, CH_SIZE)


def make_con(chs, n_samples, sfreq, *, version=2, revision=5, acq_type=1, digitize=True, fll=100, amp_data=None, adc_alloc=16, adc_stored=16, create=1772700000,
             truncate_data=0):
    nchan = len(chs)
    blocks = {}
    # system (1)
    s = struct.pack("<iii", version, revision, 52 + 0)
    s += pad(b"Invented KIT system", 128) + pad(b"Model Z", 128)
    s += struct.pack("<i", nchan) + pad(b"a comment", 256)
    s += struct.pack("<ii", create, create + 60) + b"\0" * 12
    s += struct.pack("<i", 1) + b"\0" * 12          # dewar style
    s += struct.pack("<i", fll) + b"\0" * 12        # fll type
    s += struct.pack("<i", 1) + b"\0" * 12          # trigger type
    s += struct.pack("<i", 2) + b"\0" * 116         # adboard type, reserved
    s += struct.pack("<d", 10.0) if revision > 3 else struct.pack("<i", 10)
    s += struct.pack("<iii", 1, adc_alloc, adc_stored)
    blocks[1] = s
    blocks[4] = b"".join(channel_record(c) for c in chs)
    cal = np.zeros((nchan, 2))
    for i, c in enumerate(chs):
        cal[i, 1] = c.get("gain", 1.0e-4 * (i + 1))
    blocks[5] = cal.astype("<f8").tobytes()
    if amp_data is None:
        # Kapper type (fll >= 100): gains 2, 5, 10 and the filters 3 / 2 / 1
        amp_data = (1 << 12) | (3 << 8) | (2 << 16) | (1 << 0) | (2 << 28) | (4 << 24)
    blocks[7] = struct.pack("<i", amp_data)
    a = struct.pack("<id", acq_type, sfreq)
    if acq_type == 1:
        a += struct.pack("<ii", 0, n_samples)
    else:
        a += struct.pack("<iiii", n_samples, 10, 5, 3)
    blocks[8] = a
    dig_names = ["fidnz", "fidt9", "fidt10", "hpi_1", "hpi_2", "hpi_3", "cz"]
    dig_pts = {"fidnz": [0.0, 0.1, 0.0], "fidt9": [-0.08, 0.0, 0.0], "fidt10": [0.08, 0.0, 0.0], "hpi_1": [0.05, 0.05, 0.05], "hpi_2": [-0.05, 0.05, 0.05],
               "hpi_3": [0.0, 0.0, 0.1], "cz": [0.0, 0.0, 0.11]}
    dig = b""
    n_dig = 0
    if digitize:
        for n in dig_names:
            dig += pad(n.encode(), 8) + struct.pack("<3d", *dig_pts[n])
            n_dig += 1
        for _ in range(6):
            dig += pad(b"", 8) + struct.pack("<3d", *rng.normal(size=3) * 0.05)
            n_dig += 1
        blocks[26] = dig
        c = struct.pack("<i", 1) + b"\0" * (16 * 8 * 2) + struct.pack("<i", 5)
        for m in range(5):
            c += struct.pack("<iiii", 0, 0, 1, 1 if m < 3 else 0) + struct.pack("<3d", 0, 0, 0)
            c += struct.pack("<3d", *(np.array(dig_pts[f"hpi_{min(m + 1, 3)}"]) + 0.001 * m))
        c += b"\0" * 256
        blocks[12] = c
    data = rng.integers(-3000, 3000, size=(n_samples, nchan)).astype("<i2" if adc_alloc == 16 else "<i4")
    raw = data.tobytes()
    if truncate_data:
        raw = raw[:-truncate_data]
    blocks[9] = raw
    # lay the blocks out after the directory
    offset = N_DIR * 16
    dirs = [(0, 0, 0, 0)] * N_DIR
    body = b""
    counts = {1: 1, 4: nchan, 5: nchan, 7: 1, 8: 1, 9: 1, 12: 1 if digitize else 0, 26: n_dig}
    sizes = {4: CH_SIZE}
    for k in sorted(blocks):
        dirs[k] = (offset + len(body), sizes.get(k, len(blocks[k])), counts[k], counts[k])
        body += blocks[k]
    dirs[0] = (0, 16, N_DIR, N_DIR)
    head = b"".join(struct.pack("<Iiii", *d) for d in dirs)
    return head + body, data


def named_chs():
    chs = []
    for i in range(8):
        chs.append(dict(type=2, loc=[0.05 * np.cos(i), 0.05 * np.sin(i), 0.08 + 0.002 * i, 10.0 * i, 45.0 * i], name=f"AG{i + 1:03d}" if i < 6 else "", gain=2.0e-11 * (i + 1)))
    chs += [dict(type=0x101, loc=[0.0, 0.0, 0.2, 0.0, 0.0], gain=3e-11), dict(type=0x101, loc=[0.0, 0.1, 0.2, 90.0, 0.0], gain=3e-11)]
    chs += [dict(type=-1, no=1, name="TRG001"), dict(type=-1, no=2, name="TRG002")]
    chs += [dict(type=-2, no=1, name="Cz", exg_gain=5.0e-4), dict(type=-3, no=1, name="ECG", exg_gain=1.0e-3), dict(type=-4, no=1, name="")]
    return chs


def golden_case(case, path, data):
    info, sqd = get_kit_info(path, allow_unknown_format=False)
    raw = mne.io.read_raw_kit(path, stim=None, preload=True, verbose=False)
    d = raw.get_data()
    (here / f"kit_{case}.f32").write_bytes(d.astype("<f4").tobytes())
    golden["cases"][case] = {
        "ch_names": raw.ch_names, "kinds": [int(c["kind"]) for c in raw.info["chs"]], "coil_types": [int(c["coil_type"]) for c in raw.info["chs"]],
        "units": [int(c["unit"]) for c in raw.info["chs"]], "n_times": raw.n_times, "sfreq": float(raw.info["sfreq"]),
        "nchan": int(sqd["nchan"]), "highpass": float(sqd["highpass"]), "lowpass": float(sqd["lowpass"]), "notch": float(sqd["notch"]),
        "acq_type": int(sqd["acq_type"]), "conv_factor": np.asarray(sqd["conv_factor"]).ravel().tolist(),
        "channels": [{"type": int(c["type"]), "loc": None if "loc" not in c else np.asarray(c["loc"]).tolist(), "name": c.get("name")} for c in sqd["channels"]],
        "hsp": None if "hsp" not in sqd else np.asarray(sqd["hsp"]).tolist(), "elp": None if "elp" not in sqd else np.asarray(sqd["elp"]).tolist(),
        "mrk": None if "mrk" not in sqd else np.asarray(sqd["mrk"]).tolist(), "kit_system_id": int(raw.info["kit_system_id"]),
        "description": raw.info["description"], "meas_date": raw.info["meas_date"].isoformat(), "dirs": [[int(v) for v in x.values()] for x in sqd["dirs"]],
        "eeg_dig": {k: np.asarray(v).tolist() for k, v in sqd["eeg_dig"].items()},
    }


chs = named_chs()
b, data = make_con(chs, 400, 500.0)
(root / "kit_cont.con").write_bytes(b)
golden_case("cont", root / "kit_cont.con", data)
b, data = make_con(chs, 300, 250.0, digitize=False, fll=0, amp_data=0x0000 | (1 << 11) | 2 | (3 << 4) | (2 << 8) | (1 << 14), adc_alloc=32, adc_stored=32)
(root / "kit_nodig_i32.con").write_bytes(b)
golden_case("nodig_i32", root / "kit_nodig_i32.con", data)

# refusals
(root / "kit_old.con").write_bytes(make_con(chs, 50, 500.0, version=1, revision=5)[0])
(root / "kit_epoched.con").write_bytes(make_con(chs, 50, 500.0, acq_type=3)[0])
(root / "kit_evoked.con").write_bytes(make_con(chs, 50, 500.0, acq_type=2)[0])
(root / "kit_truncated.con").write_bytes(make_con(chs, 400, 500.0, digitize=False, truncate_data=200)[0])
(root / "kit_not.con").write_bytes(b"not a KIT file" * 40)

def clean(x):
    """plain JSON: numpy scalars as numbers, NaN and infinity as null"""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


with open(here / "kit_golden.json", "w") as f:
    json.dump(clean(golden), f, separators=(",", ":"))
size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
print("kit data", size, "bytes;", "golden", (here / "kit_golden.json").stat().st_size)
