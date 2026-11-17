#!/usr/bin/env python3
"""A synthetic source estimate written by MNE-Python (a reference for parity only): test/data/stc_basic-lh.stc, stc_basic-rh.stc and test/golden/stc_golden.json.

The estimate: left vertices 0, 2, 3, 7, 10 and right vertices 1, 4, 5; 4 time points from 0.25 s every 0.125 s; value = vertex + 100 * hemisphere + 10 * time index
(hemisphere 0 left, 1 right). Run:
  uv run --project <nxr> python test/golden/gen_stc.py
"""
import json
import pathlib

import mne
import numpy as np

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
data_dir.mkdir(exist_ok=True)

left, right = np.array([0, 2, 3, 7, 10]), np.array([1, 4, 5])
data = np.array([[v + 10.0 * t for t in range(4)] for v in left] + [[v + 100.0 + 10.0 * t for t in range(4)] for v in right], dtype=np.float32)   # vertices x times
stc = mne.SourceEstimate(data, [left, right], tmin=0.25, tstep=0.125, subject="synthetic")
stc.save(str(data_dir / "stc_basic"), ftype="stc", overwrite=True)

back = mne.read_source_estimate(str(data_dir / "stc_basic-lh.stc"), subject="synthetic")
(here / "stc_golden.json").write_text(json.dumps({
    "tmin": float(back.tmin), "tstep": float(back.tstep), "left": [int(v) for v in back.vertices[0]], "right": [int(v) for v in back.vertices[1]],
    "data": back.data.astype(float).tolist(), "mne": mne.__version__}, indent=1))
print("ok")
