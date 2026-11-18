"""Writes montage_golden.txt: MNE-Python's `spherical_1005` standard montage (mne.channels.make_standard_montage), the reference of nxr::meeg::tenFive on a SPHERE (MNE's file is the
10-05 rule of Oostenveld and Praamstra 2001 computed on a sphere of radius 0.095 m, from the `eeg_positions` package). The generator only CALLS MNE; its output is the golden: the 10-05 labels with
their positions (metres, MNE's head frame: x to the right, y to the front, z up; the landmarks NAS, LPA, RPA included, and Nz, T9, T10 which coincide with them) and the label lists of MNE's
`spherical_1010` and `spherical_1020`. Nothing from a real head.
Run:  uv run --with mne python compute/test/golden/gen_montage_golden.py compute/test/golden/montage_golden.txt"""
import sys

import mne

out = sys.argv[1]
m = mne.channels.make_standard_montage("spherical_1005")
pos = m.get_positions()
ch = pos["ch_pos"]
lines = [f"# MNE-Python {mne.__version__} make_standard_montage('spherical_1005'), metres, head frame (x right, y front, z up)"]
lines.append(f"landmark NAS {' '.join('%.9g' % v for v in pos['nasion'])}")
lines.append(f"landmark LPA {' '.join('%.9g' % v for v in pos['lpa'])}")
lines.append(f"landmark RPA {' '.join('%.9g' % v for v in pos['rpa'])}")
for name, p in ch.items():
    lines.append(f"point {name} {' '.join('%.9g' % v for v in p)}")
for system in ("1010", "1020"):
    names = mne.channels.make_standard_montage(f"spherical_{system}").ch_names
    lines.append(f"system {system} {' '.join(names)}")
open(out, "w").write("\n".join(lines) + "\n")
print(len(ch), "points,", out)
