"""Two SYNTHETIC Brainstorm protocols for the wasm-io import test (no real data), with the importer parity tests' own builders (io/test/parity/convert_fixtures.py):
    session/  S01: an octahedron cortex, a channel file and a raw EEG recording (condition COND) — `convert convert`
    full/     the full protocol: subjects S01, S02 and the default anatomy (with an MRI) — `convert template`
    NXR=<nxr checkout> uv run --project $NXR python make_protocols.py <out>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "io" / "test" / "parity"))
import convert_fixtures as cf  # noqa: E402
from test_sensors_mri_parity import build_images  # noqa: E402

out = Path(sys.argv[1])
# three EEG electrodes on a 9 cm sphere (the fixture's default puts every one at the origin, where no head sphere can be fitted)
import numpy as np  # noqa: E402
locs = [(0.09, 0.0, 0.0), (0.0, 0.09, 0.0), (0.0, 0.0, 0.09)]
chan = {"Comment": "EEG channels", "Channel": np.array([(f"EEG{i:03d}", "EEG", np.array(xyz).reshape(3, 1), np.array([]), np.array([])) for i, xyz in enumerate(locs)],
                                                        dtype=[("Name", "O"), ("Type", "O"), ("Loc", "O"), ("Orient", "O"), ("Weight", "O")])}
cf.build_session_protocol(out / "session", condition="sub-S01_ses-01_task-rest_run-01_meg", n_chan=3, n_t=1000, channel_mat=chan)
cf.build_protocol(out / "full")
build_images(out / "full" / "anat" / "@default_subject")   # the default anatomy's MRI: a template is imported with its subject image
print(out)
