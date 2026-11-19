#!/usr/bin/env python3
"""Write a small SYNTHETIC EEG dataset in BIDS form, and the meshes of its head, for the tutorials.

    python3 synthetic_bids.py OUTPUT_FOLDER

Nothing here is real data. A 10 Hz source that comes in eight bursts (a current dipole at a known place on a spherical 'cortex') plus a weaker steady 20 Hz source elsewhere and
white noise are projected onto 32 electrodes on a spherical 'scalp' (the potential of a current dipole in an unbounded conductor), and written as
EDF with the BIDS tables (channels, electrodes, coordinate system). The folder also gets cortex.obj and scalp.obj (spheres, in metres) and truth.json
(where the sources are), so a tutorial can check what the toolbox recovers against what was put in. Needs NumPy only.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

SFREQ = 250
SECONDS = 60
N_ELECTRODES = 32
CORTEX_RADIUS = 0.07   # m
SCALP_RADIUS = 0.10    # m


def icosphere(radius, subdivisions):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdivisions):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v) * radius, np.array(f)


def write_obj(path, v, f):
    with open(path, "w") as h:
        for p in v:
            h.write("v %.9f %.9f %.9f\n" % tuple(p))
        for a, b, c in f:
            h.write("f %d %d %d\n" % (a + 1, b + 1, c + 1))


def write_edf(path, x_uv, names, sfreq):
    """EDF, one second per data record, int16 samples; x_uv is [channels, samples] in microvolts."""
    ns, n = x_uv.shape
    nrec = n // sfreq
    pmax = float(np.ceil(np.abs(x_uv).max() * 1.05))
    def f(s, k): return str(s).ljust(k)[:k].encode()
    head = f("0", 8) + f("X X X X", 80) + f("Startdate 01-JAN-2000 X X X", 80) + f("01.01.00", 8) + f("00.00.00", 8) + f(256 + 256 * ns, 8) + f("", 44) + f(nrec, 8) + f("1", 8) + f(ns, 4)
    sig = b"".join(f(s, 16) for s in names) + b"".join(f("", 80) for _ in names) + b"".join(f("uV", 8) for _ in names)
    sig += b"".join(f(-pmax, 8) for _ in names) + b"".join(f(pmax, 8) for _ in names) + b"".join(f(-32768, 8) for _ in names) + b"".join(f(32767, 8) for _ in names)
    sig += b"".join(f("", 80) for _ in names) + b"".join(f(sfreq, 8) for _ in names) + b"".join(f("", 32) for _ in names)
    q = np.round(x_uv / pmax * 32767).astype("<i2")
    with open(path, "wb") as h:
        h.write(head + sig)
        for r in range(nrec):
            for c in range(ns):
                h.write(q[c, r * sfreq:(r + 1) * sfreq].tobytes())


def dipole_potential(sensors, position, moment):
    d = sensors - position
    return d @ moment / (4 * np.pi * np.linalg.norm(d, axis=1) ** 3)


def main(out):
    out = Path(out)
    rng = np.random.default_rng(7)
    eeg = out / "bids" / "sub-01" / "eeg"
    eeg.mkdir(parents=True, exist_ok=True)

    cortex_v, cortex_f = icosphere(CORTEX_RADIUS, 3)
    scalp_v, scalp_f = icosphere(SCALP_RADIUS, 2)
    write_obj(out / "cortex.obj", cortex_v, cortex_f)
    write_obj(out / "scalp.obj", scalp_v, scalp_f)

    # electrodes: a Fibonacci spiral over the upper two thirds of the scalp sphere
    k = np.arange(N_ELECTRODES) + 0.5
    z = 1 - 1.2 * k / N_ELECTRODES
    phi = np.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z ** 2)
    sensors = SCALP_RADIUS * np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    names = ["E%02d" % (i + 1) for i in range(N_ELECTRODES)]

    # two sources on the cortex: 10 Hz (strong) and 20 Hz (weaker), tangential current dipoles
    def tangential(p):
        a = np.cross(p, [0, 0, 1.0])
        return a / np.linalg.norm(a)
    i10 = int(np.argmin(np.linalg.norm(cortex_v - CORTEX_RADIUS * np.array([0.6, 0.2, 0.77]) / np.linalg.norm([0.6, 0.2, 0.77]), axis=1)))
    i20 = int(np.argmin(np.linalg.norm(cortex_v - CORTEX_RADIUS * np.array([-0.7, -0.3, 0.64]) / np.linalg.norm([-0.7, -0.3, 0.64]), axis=1)))
    t = np.arange(SFREQ * SECONDS) / SFREQ
    q = 1e-8  # A m
    # the 10 Hz source comes in bursts: a Hann-shaped envelope 1.5 s long at known times on a weak steady background
    burst_starts = [4.0, 11.5, 19.0, 27.0, 33.5, 41.0, 48.0, 54.0]
    envelope = np.full(t.shape, 0.15)
    for b in burst_starts:
        inside = (t >= b) & (t < b + 1.5)
        envelope[inside] += np.hanning(inside.sum())
    x = (np.outer(dipole_potential(sensors, cortex_v[i10], q * tangential(cortex_v[i10])), envelope * np.sin(2 * np.pi * 10 * t))
         + np.outer(dipole_potential(sensors, cortex_v[i20], 0.5 * q * tangential(cortex_v[i20])), np.sin(2 * np.pi * 20 * t + 1.0)))
    x_uv = (x + rng.normal(0, 0.15 * x.std(), x.shape)) * 1e6

    write_edf(eeg / "sub-01_task-rest_eeg.edf", x_uv, names, SFREQ)
    (eeg / "sub-01_task-rest_eeg.json").write_text(json.dumps({"TaskName": "rest", "SamplingFrequency": SFREQ, "EEGReference": "n/a", "EEGChannelCount": N_ELECTRODES, "PowerLineFrequency": 50}))
    (eeg / "sub-01_task-rest_channels.tsv").write_text("name\ttype\tunits\tsampling_frequency\tstatus\n" + "".join("%s\tEEG\tuV\t%d\tgood\n" % (n, SFREQ) for n in names))
    (eeg / "sub-01_electrodes.tsv").write_text("name\tx\ty\tz\n" + "".join("%s\t%.6f\t%.6f\t%.6f\n" % ((n,) + tuple(p)) for n, p in zip(names, sensors)))
    (eeg / "sub-01_coordsystem.json").write_text(json.dumps({"EEGCoordinateSystem": "Other", "EEGCoordinateUnits": "m", "EEGCoordinateSystemDescription": "synthetic head sphere centred on the origin"}))
    (out / "bids" / "dataset_description.json").write_text(json.dumps({"Name": "synthetic tutorial data", "BIDSVersion": "1.9.0", "DatasetType": "raw"}))
    (out / "bids" / "participants.tsv").write_text("participant_id\tage\nsub-01\t30\n")
    (out / "truth.json").write_text(json.dumps({
        "sources": [{"frequency_hz": 10, "vertex": i10, "position_m": cortex_v[i10].tolist()}, {"frequency_hz": 20, "vertex": i20, "position_m": cortex_v[i20].tolist()}],
        "bursts_10hz": [[b, b + 1.5] for b in burst_starts], "sfreq": SFREQ, "seconds": SECONDS, "electrodes": N_ELECTRODES, "cortex_vertices": len(cortex_v)}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
