"""Parity of the channel file's projectors and CTF compensation (register D168): `nxr-neuro` does what nxr's Python does, on SYNTHETIC files only.

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro uv run --project $NXR python -m pytest -q test/parity/test_projectors_parity.py

Both tools import a session whose channel file carries SSP and ICA projectors (status 1 · 2 · 0) and a MegRefCoef; every row, every zarr.json and every chunk's
bytes are equal — the operators, and the samples with the status-1 projectors applied (raw link and imported excerpt, native rate and resampled).

With NXR_IO_BST_PROTOCOL, NXR_IO_BST_SUBJECT and NXR_IO_BST_CHANNEL (a real channel file of ONE subject, read only) set, `test_a_real_channel_file_*` runs the same on it
with synthetic samples. Unset, it skips.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
IO = Path(os.environ["NXR_NEURO"]) if "NXR_NEURO" in os.environ else None
pytestmark = pytest.mark.skipif(NXR is None or IO is None, reason="set NXR and NXR_NEURO")

if NXR is not None:
    import convert_fixtures as cf  # noqa: E402
    from test_convert_parity import assert_same_dataset, lines, run_io, run_nxr, starting_pair  # noqa: E402

COND = "sub-S01_task-rest_run-01"
TYPES = ["MEG", "MEG", "MEG", "MEG", "MEG REF", "MEG REF", "EEG", "ECG"]
M = len(TYPES)


def structs(rows):
    keys = ["Comment", "Components", "CompMask", "Status", "SingVal", "Method"]
    return np.array([tuple(r[k] for k in keys) for r in rows], dtype=[(k, "O") for k in keys])


def ortho(k, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((M, M)))
    return q[:, :k]


def projector(status, comp, mask, comment, method="SSP_pca"):
    return {"Comment": comment, "Components": comp, "CompMask": np.array(mask, dtype=np.uint8), "Status": float(status), "SingVal": np.ones(comp.shape[1]), "Method": method}


def channel_file(path: Path, projectors, coef=True) -> Path:
    chan = np.array([(f"C{i}", TYPES[i], np.array([[0.01 * i], [0.0], [0.05]]), np.array([[1.0], [0.0], [0.0]]), np.ones(1), "") for i in range(M)],
                    dtype=[("Name", "O"), ("Type", "O"), ("Loc", "O"), ("Orient", "O"), ("Weight", "O"), ("Group", "O")])
    extra = {"Projector": structs(projectors)} if projectors else {}
    if coef:
        extra["MegRefCoef"] = np.random.default_rng(1).standard_normal((4, 2))
    savemat(str(path), {"Comment": "synthetic", "Channel": chan, **extra})
    return path


def all_projectors():
    w = np.random.default_rng(5).standard_normal((3, M))
    return [projector(1, ortho(3, 1), [1, 1, 0], "ssp active"),
            projector(2, ortho(2, 2), [1, 0], "ssp applied"),
            projector(0, ortho(2, 3), [0, 1], "ssp inactive"),
            projector(1, w.T, [0, 1, 1], "ica active", "ICA_infomax")]


def session(tmp: Path, channel: Path, *, imported: bool, grade=3) -> Path:
    m = loadmat(str(channel))
    n = int(np.size(m["Channel"]))
    root = tmp / "bst"
    if imported:
        F = cf.session_data(n, 700, 3)
        cf.build_session_protocol(root, condition=COND, n_chan=n, n_t=700, epoch=256, raw=False, imported=[("exc", F, {})])
        dest = root / "data" / "S01" / COND
    else:
        cf.build_session_protocol(root, condition=COND, n_chan=n, n_t=1500, epoch=256)
        dest = root / "data" / "S01" / f"@raw{COND}"
        link = loadmat(str(dest / f"data_0raw_{COND}.mat"), squeeze_me=True, struct_as_record=False)["F"]
        d = loadmat(str(dest / f"data_0raw_{COND}.mat"), squeeze_me=False)
        f = d["F"]
        f["prop"][0, 0]["currCtfComp"][0, 0] = np.array([[grade]])
        f["prop"][0, 0]["destCtfComp"][0, 0] = np.array([[grade]])
        savemat(str(dest / f"data_0raw_{COND}.mat"), {k: v for k, v in d.items() if not k.startswith("__")})
    import shutil
    shutil.copy(channel, dest / "channel.mat")
    return root


def run_both(tmp: Path, root: Path, *extra):
    a, b = starting_pair(tmp)
    args = ["convert", "convert", root, "--subject", "S01", "--condition", COND, *extra]
    pa = run_nxr(*args, "--dataset", a, start=1000)
    assert pa.returncode == 0, pa.stdout[-2000:] + pa.stderr[-2000:]
    pb = run_io(*args, "--dataset", b, start=1000)
    assert pb.returncode == 0, pb.stdout + pb.stderr
    assert_same_dataset(a, b)
    assert lines(pa) == lines(pb)
    return a, b, lines(pb)


@pytest.mark.parametrize("imported", [False, True], ids=["raw-link", "excerpt"])
@pytest.mark.parametrize("resample", [None, "50"], ids=["native", "resampled"])
def test_projectors_are_nxrs(imported, resample, tmp_path):
    root = session(tmp_path, channel_file(tmp_path / "channel.mat", all_projectors()), imported=imported)
    a, b, ls = run_both(tmp_path, root, *(["--resample", resample] if resample else []))
    assert [x["stage"] for x in ls if x["stage"] == "projectors"] == ["projectors"]
    con = sqlite3.connect(str(b / "dataset.sqlite"))
    ops = {r[0]: r[1:] for r in con.execute("SELECT name, kind, layout, params_json FROM operator WHERE kind IN ('projector', 'ctf compensation') ORDER BY name")}
    assert sorted(ops) == ["channel_ctf_compensation", "channel_projector_1", "channel_projector_2", "channel_projector_3", "channel_projector_4"]
    assert [json.loads(ops[f"channel_projector_{k}"][2])["status"] for k in (1, 2, 3, 4)] == [1, 2, 0, 1]
    assert json.loads(ops["channel_projector_4"][2])["kind"] == "ica"
    prod = json.loads(con.execute("SELECT producer_json FROM field WHERE kind = 'recording'").fetchone()[0])
    assert [p["applied"] for p in prod["projectors"]] == ["applied by the import", "already applied to the samples", "not applied", "applied by the import"]
    assert prod["ctf_compensation"]["operator"] == "channel_ctf_compensation"
    assert ("grade" in prod["ctf_compensation"]) == (not imported)


def test_no_projector_no_operators_and_the_samples_are_the_files(tmp_path):
    root = session(tmp_path, channel_file(tmp_path / "channel.mat", [], coef=False), imported=False)
    _, b, ls = run_both(tmp_path, root)
    con = sqlite3.connect(str(b / "dataset.sqlite"))
    assert con.execute("SELECT count(*) FROM operator WHERE kind IN ('projector', 'ctf compensation')").fetchone()[0] == 0
    assert "projectors" not in [x["stage"] for x in ls]
    assert "projectors" not in json.loads(con.execute("SELECT producer_json FROM field WHERE kind = 'recording'").fetchone()[0])


def test_only_status_two_and_zero_leave_the_samples(tmp_path):
    ps = [projector(2, ortho(2, 2), [1, 0], "applied"), projector(0, ortho(2, 3), [0, 1], "inactive")]
    root = session(tmp_path, channel_file(tmp_path / "channel.mat", ps, coef=False), imported=False)
    run_both(tmp_path, root)


REAL = [os.environ.get(k) for k in ("NXR_IO_BST_PROTOCOL", "NXR_IO_BST_SUBJECT", "NXR_IO_BST_CHANNEL")]


@pytest.mark.skipif(not all(REAL), reason="set NXR_IO_BST_PROTOCOL, NXR_IO_BST_SUBJECT, NXR_IO_BST_CHANNEL")
def test_a_real_channel_file_is_nxrs(tmp_path):
    m = loadmat(REAL[2])
    n = int(np.size(m["Channel"]))
    root = tmp_path / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=n, n_t=1500, epoch=256)
    import shutil
    shutil.copy(REAL[2], root / "data" / "S01" / f"@raw{COND}" / "channel.mat")
    run_both(tmp_path, root)
