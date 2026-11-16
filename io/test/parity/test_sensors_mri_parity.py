"""Parity of the sensors (register D166) and the MRI importers: `nxr-neuro` does what nxr's Python does, on SYNTHETIC files only.

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro uv run --project $NXR python -m pytest -q test/parity/test_sensors_mri_parity.py

The same call runs through nxr's Python and through nxr-neuro, ids a counter and a fixed clock on both sides, into two copies of one starting dataset; every row
of every table, every zarr.json (as text) and every chunk file's bytes are equal. The sensors have no command of their own in nxr (the session import calls
them), so both tools import a SESSION whose channel file is the one under test (`convert convert`; a raw link of that many channels).

With NXR_IO_BST_PROTOCOL and NXR_IO_BST_SUBJECT set to a Brainstorm protocol and ONE subject of it (read only), `test_a_real_subject_*` compares the two on its
anatomy, its images and one channel file (NXR_IO_BST_CHANNEL, a path under the protocol); the outputs go under NXR_IO_BST_OUT. Unset, they skip.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

NXR = Path(os.environ["NXR"]) if "NXR" in os.environ else None
IO = Path(os.environ["NXR_NEURO"]) if "NXR_NEURO" in os.environ else None
pytestmark = pytest.mark.skipif(NXR is None or IO is None, reason="set NXR and NXR_NEURO")

if NXR is not None:
    import convert_fixtures as cf  # noqa: E402,F401
    from test_convert_parity import assert_same_dataset, lines, run_io, run_nxr, starting_pair  # noqa: E402
    import writer_scenario as ws  # noqa: E402

COND = "sub-S01_task-rest_run-01"


def session_with(tmp: Path, channel: Path) -> Path:
    """a protocol whose one session has this channel file (and the file's flags on its raw link): the anatomy, a link and a .bst of that many channels"""
    import shutil
    from scipy.io import loadmat
    m = loadmat(str(channel))
    n = int(np.size(m["Channel"]))
    flags = m["ChannelFlag"].ravel() if "ChannelFlag" in m else None
    root = tmp / "bst"
    cf.build_session_protocol(root, condition=COND, n_chan=n, n_t=600, epoch=256, flags=flags)
    shutil.copy(channel, root / "data" / "S01" / f"@raw{COND}" / "channel.mat")
    return root


def channel_file(path: Path, rows, **extra) -> Path:
    chan = np.array([(n, t, np.asarray(l, float), np.asarray(o, float), np.asarray(w, float), g) for n, t, l, o, w, g in rows],
                    dtype=[("Name", "O"), ("Type", "O"), ("Loc", "O"), ("Orient", "O"), ("Weight", "O"), ("Group", "O")])
    savemat(str(path), {"Comment": "synthetic", "Channel": chan, **extra})
    return path


def cell(*items):
    c = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        c[i] = x
    return c


def tf(k):
    m = np.eye(4)
    m[:3, 3] = [k, 2 * k, 3 * k]
    return m


Z = np.zeros((0, 0))


def full_channels(tmp: Path) -> Path:
    c1 = np.c_[[0.1, 0, 0], [0.11, 0, 0], [0.12, 0, 0], [0.13, 0, 0]]
    c2 = np.c_[[0, 0.1, 0], [0, 0.11, 0], [0, 0.12, 0], [0, 0.13, 0]]
    return channel_file(tmp / "channel_full.mat", [
        ("MLC11", "MEG", c1, np.c_[[1, 0, 0]] * np.ones(4), [0.25] * 4, ""),
        ("MRC11", "MEG", c2, np.c_[[0, 1, 0]] * np.ones(4), [0.5, 0.25, 0.125, 0.125], ""),
        ("MZO01", "MEG", np.c_[[0.2, 0.2, 0]], np.c_[[0, 0, 1]], [1.0], ""),
        ("BG1", "MEG REF", np.c_[[0.3, 0.3, 0.3]], np.c_[[0, 0, 1]], [1.0], ""),
        ("Cz", "EEG", np.c_[[0, 0, 0.09]], Z, Z, ""),
        ("A1-A2", "SEEG", np.c_[[0.01, 0, 0], [0.02, 0, 0]], Z, Z, "A"),
        ("ECG", "ECG", Z, Z, Z, ""),
    ], ChannelFlag=np.array([[1, 1, -1, 1, 1, 1, 1.0]]),
        HeadPoints=np.array([([[0.0, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]], np.array(["NAS", "LPA", "EXTRA"], dtype=object),
                              np.array(["CARDINAL", "CARDINAL", "EXTRA"], dtype=object))], dtype=[("Loc", "O"), ("Label", "O"), ("Type", "O")]),
        TransfMeg=cell(tf(1), tf(2)), TransfMegLabels=np.array(["Native=>Brainstorm/CTF", "Adjust"], dtype=object),
        TransfEeg=cell(tf(3)), TransfEegLabels=np.array(["Refine"], dtype=object),
        SCS={"NAS": np.array([0.1, 0, 0]), "LPA": np.array([0, 0.07, 0]), "RPA": np.array([0, -0.07, 0]), "R": np.eye(3), "T": np.array([0.0, 0.0, 0.01])})


def plain_channels(tmp: Path) -> Path:
    return channel_file(tmp / "channel_eeg.mat", [("Cz", "EEG", np.c_[[0, 0, 0.09]], Z, Z, ""), ("Pz", "EEG", np.c_[[0, -0.05, 0.08]], Z, Z, ""),
                                                  ("EOG", "EOG", Z, Z, Z, "")])


def seeg_channels(tmp: Path) -> Path:
    return channel_file(tmp / "channel_seeg.mat", [("A1-A2", "SEEG", np.c_[[0.01, 0, 0], [0.02, 0, 0]], Z, Z, "A"),
                                                   ("B1-B2", "SEEG", np.c_[[0, 0.03, 0], [0, 0.05, 0]], Z, Z, "B'"),
                                                   ("B3", "SEEG", np.c_[[0, 0, 0.07]], Z, Z, "B'")])


@pytest.mark.parametrize("maker", [full_channels, plain_channels, seeg_channels])
def test_sensors_are_nxrs(maker, tmp_path):
    root = session_with(tmp_path, maker(tmp_path))
    a, b = starting_pair(tmp_path)
    args = ["convert", "convert", root, "--subject", "S01", "--condition", COND]
    pa = run_nxr(*args, "--dataset", a, start=1000)
    assert pa.returncode == 0, pa.stdout[-2000:] + pa.stderr[-2000:]
    pb = run_io(*args, "--dataset", b, start=1000)
    assert pb.returncode == 0, pb.stdout + pb.stderr
    assert_same_dataset(a, b)
    assert lines(pa) == lines(pb) and [x["stage"] for x in lines(pb)][:2] == ["surface", "sensors"]


def test_the_full_geometry_is_there(tmp_path):
    root = session_with(tmp_path, full_channels(tmp_path))
    _, b = starting_pair(tmp_path)
    run_io("convert", "convert", root, "--subject", "S01", "--condition", COND, "--dataset", b, ok=True, start=1000)
    import sqlite3
    con = sqlite3.connect(str(b / "dataset.sqlite"))
    paths = {r[0] for r in con.execute("SELECT path FROM manifold UNION SELECT path FROM field UNION SELECT path FROM operator UNION SELECT path FROM selection")}
    for p in ("sensor/channel", "sensor/channel_orientation", "sensor/channel_coils", "sensor/channel_coils_orientation", "channel_coil_weights",
              "sensor/channel_headpoints", "sensor/channel_headpoints_labels/label", "sensor/channel_headpoints_labels/type", "sensor/channel_contact1"):
        assert p in paths, p
    pj = json.loads(con.execute("SELECT producer_json FROM manifold WHERE path = 'sensor/channel'").fetchone()[0])
    assert set(pj["transforms"]) == {"meg", "eeg", "scs"}


# ── MRI ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def frame(voxsize, rotated=True):
    """a valid SCS for a 16^3 grid: scs_mm = R @ (voxel * voxsize) + T, the origin at voxel (8, 8, 8), fiducials on the z = 0 plane"""
    R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) if rotated else np.eye(3)
    # the x axis (anterior) along voxel j when rotated: scs = (v_j, -v_i, v_k)
    origin = np.array([8.0, 8.0, 8.0])
    T = -R @ (origin * voxsize)
    inv = lambda p: (R.T @ (np.asarray(p, float) - T)) / voxsize
    return {"R": R, "T": T, "NAS": inv([10, 0, 0]), "LPA": inv([0, 7, 0]), "RPA": inv([0, -7, 0]), "Origin": origin}


def write_image(path: Path, cube, *, comment, voxsize=(1.0, 1.0, 1.5), labels=None, ncs=True, origin=True, scs=None, compress=True):
    vs = np.asarray(voxsize, float)
    s = scs or frame(vs)
    if not origin:
        s = {k: v for k, v in s.items() if k != "Origin"}
    d = {"Cube": cube, "Voxsize": vs.reshape(1, 3), "SCS": s, "Comment": comment,
         "Labels": labels if labels is not None else np.zeros((0, 0))}
    if ncs:
        d["NCS"] = {"R": np.eye(3) * 0.9, "T": np.array([1.0, 2.0, 3.0]), "AC": np.array([1.0, 2, 3]), "PC": np.array([4.0, 5, 6]), "IH": np.array([7.0, 8, 9])}
    savemat(str(path), d, do_compression=compress)
    return path


def labels_cell(rows):
    c = np.empty((len(rows), 3), dtype=object)
    for i, (v, n, rgb) in enumerate(rows):
        c[i, 0], c[i, 1], c[i, 2] = np.array([[float(v)]]), n, np.array([rgb], dtype=float)
    return c


def build_images(anat: Path, *, compress=True, extra=True) -> None:
    anat.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    t1 = rng.integers(0, 256, size=(16, 16, 16)).astype(np.uint8)
    write_image(anat / "subjectimage_MRI_T1.mat", t1, comment="MRI: T1", compress=compress)
    atlas = rng.choice(np.array([0, 2, 3, 41, 42, 77], dtype=np.uint8), size=(16, 16, 16)).astype(np.uint8)       # 77 is not in the table
    write_image(anat / "subjectimage_ASEG_volatlas.mat", atlas, comment="ASEG", ncs=False, origin=False, compress=compress,
                labels=labels_cell([(0, "Unknown", [0, 0, 0]), (2, "Left-Cerebral-White-Matter", [245, 245, 245]), (3, "Left-Cerebral-Cortex", [205, 62, 78]),
                                    (41, "Right-Cerebral-White-Matter", [245, 245, 245]), (42, "Right-Cerebral-Cortex", [205, 62, 78])]))
    if extra:
        pet = rng.random((16, 16, 16)).astype(np.float32) * 3
        write_image(anat / "subjectimage_260928_1516.mat", pet, comment="PET 18FNAV4694_suvr", ncs=False, compress=compress)
        big = rng.integers(0, 300, size=(16, 16, 16)).astype(np.uint16)
        write_image(anat / "subjectimage_Destrieux_volatlas.mat", big, comment="Destrieux", ncs=False, compress=compress,
                    labels=labels_cell([(i, f"label {i}", [i % 256, (i * 3) % 256, 7]) for i in range(300)]))


@pytest.mark.parametrize("compress", [True, False], ids=["v7", "v6"])
def test_convert_mri_is_nxrs(compress, tmp_path):
    anat = tmp_path / "anat" / "S01"
    build_images(anat, compress=compress)
    a, b = starting_pair(tmp_path)
    pa = run_nxr("convert", "mri", "--dataset", a, "--subject", "S01", "--anat", anat, start=1000)
    pb = run_io("convert", "mri", "--dataset", b, "--subject", "S01", "--anat", anat, start=1000)
    assert pa.returncode == pb.returncode == 0, pa.stdout + pa.stderr + pb.stdout + pb.stderr
    assert_same_dataset(a, b)
    la, lb = lines(pa), lines(pb)
    assert la == lb, (la, lb)
    assert [x["kind"] for x in lb if x["stage"] == "mri"] == ["anatomical", "atlas", "atlas", "scalar"]


def test_only_selects_by_name(tmp_path):
    anat = tmp_path / "anat" / "S01"
    build_images(anat)
    a, b = starting_pair(tmp_path)
    pa = run_nxr("convert", "mri", "--dataset", a, "--subject", "S01", "--anat", anat, "--only", "T1", "ASEG", start=1000)
    pb = run_io("convert", "mri", "--dataset", b, "--subject", "S01", "--anat", anat, "--only", "T1", "ASEG", start=1000)
    assert pa.returncode == pb.returncode == 0
    assert_same_dataset(a, b)
    assert lines(pa) == lines(pb)


def test_a_mirrored_volume_is_refused_and_reported_like_nxr(tmp_path):
    anat = tmp_path / "anat" / "S01"
    anat.mkdir(parents=True)
    vs = np.array([1.0, 1.0, 1.5])
    s = frame(vs)
    flip = np.diag([1.0, -1.0, 1.0])
    s["R"], s["T"] = flip @ s["R"], flip @ s["T"]
    write_image(anat / "subjectimage_MRI_T1.mat", np.zeros((16, 16, 16), np.uint8), comment="T1", scs=s)
    a, b = starting_pair(tmp_path)
    pa = run_nxr("convert", "mri", "--dataset", a, "--subject", "S01", "--anat", anat)
    pb = run_io("convert", "mri", "--dataset", b, "--subject", "S01", "--anat", anat)
    assert pa.returncode == pb.returncode == 1
    ma, mb = lines(pa)[-1]["message"], lines(pb)[-1]["message"]
    assert "LEFT-RIGHT ORIENTATION IS WRONG" in mb
    assert ma.replace("FileNotFoundError", "") == mb.replace("FileNotFoundError", ""), (ma, mb)


def test_an_atlas_without_its_t1_and_other_refusals_are_reported_like_nxr(tmp_path):
    anat = tmp_path / "anat" / "S01"
    build_images(anat, extra=False)
    (anat / "subjectimage_MRI_T1.mat").unlink()
    write_image(anat / "subjectimage_ct.mat", np.zeros((16, 16, 16), np.int16), comment="CT")           # int16: refused
    a, b = starting_pair(tmp_path)
    pa = run_nxr("convert", "mri", "--dataset", a, "--subject", "S01", "--anat", anat)
    pb = run_io("convert", "mri", "--dataset", b, "--subject", "S01", "--anat", anat)
    assert pa.returncode == pb.returncode == 1
    assert lines(pa) == lines(pb), (pa.stdout, pb.stdout)


def test_a_grid_that_is_not_the_t1s_is_refused(tmp_path):
    anat = tmp_path / "anat" / "S01"
    build_images(anat, extra=False)
    write_image(anat / "subjectimage_other.mat", np.zeros((16, 16, 16), np.float32), comment="other", voxsize=(1.0, 1.0, 2.0), ncs=False)
    a, b = starting_pair(tmp_path)
    pa = run_nxr("convert", "mri", "--dataset", a, "--subject", "S01", "--anat", anat, start=1000)
    pb = run_io("convert", "mri", "--dataset", b, "--subject", "S01", "--anat", anat, start=1000)
    assert pa.returncode == pb.returncode == 0
    assert any(x["stage"] == "skipped" and "differs" in x["reason"] for x in lines(pb))
    assert lines(pa) == lines(pb)
    assert_same_dataset(a, b)


def test_convert_subject_with_surfaces_and_images_is_nxrs(tmp_path):
    root = tmp_path / "proto"
    cf.build_protocol(root, kernels=False)
    build_images(root / "anat" / "S01")
    a, b = starting_pair(tmp_path, subjects=())
    pa = run_nxr("convert", "subject", root, "--subject", "S01", "--dataset", a, start=1000)
    assert pa.returncode == 0, pa.stdout[-2000:] + pa.stderr[-2000:]
    pb = run_io("convert", "subject", root, "--subject", "S01", "--dataset", b, start=1000)
    assert pb.returncode == 0, pb.stdout + pb.stderr
    assert_same_dataset(a, b)
    out = lines(pb)
    assert [x["name"] for x in out if x["stage"] == "mri"] == ["T1", "ASEG", "Destrieux", "PET_18FNAV4694_suvr"]
    assert lines(pb)[-1]["volumes"] == ["T1", "ASEG", "Destrieux", "PET_18FNAV4694_suvr"]


def test_a_v73_shaped_file_without_a_cube_is_refused_with_its_name(tmp_path):
    """changed meaning: was `test_a_v73_image_is_refused_with_its_name` ("v7.3" in the message). `write_v73` makes an HDF5 file with MATLAB's header and no
    MATLAB variables, so the file opens but holds no Cube; a genuine v7.3 file is read like its v7 twin (test_convert_parity's v7.3 surface and kernel)."""
    anat = tmp_path / "anat" / "S01"
    anat.mkdir(parents=True)
    cf.write_v73(anat / "subjectimage_MRI_T1.mat")
    _, b = starting_pair(tmp_path)
    p = run_io("convert", "mri", "--dataset", b, "--subject", "S01", "--anat", anat)
    assert p.returncode == 1 and "no Cube" in lines(p)[-1]["message"] and "subjectimage_MRI_T1.mat" in lines(p)[-1]["message"]


# ── a real subject (optional): the environment names it, nothing here does ─────────────────────────────────────────────

REAL = {k: os.environ.get(k) for k in ("NXR_IO_BST_PROTOCOL", "NXR_IO_BST_SUBJECT", "NXR_IO_BST_OUT")}
real_only = pytest.mark.skipif(not all(REAL.values()), reason="set NXR_IO_BST_PROTOCOL, NXR_IO_BST_SUBJECT (ONE subject, read only) and NXR_IO_BST_OUT")


def real_pair(kind: str) -> tuple[Path, Path]:
    out = Path(REAL["NXR_IO_BST_OUT"]) / kind
    out.mkdir(parents=True, exist_ok=True)
    return starting_pair(out)


@real_only
def test_a_real_subject_anatomy_and_images_are_nxrs():
    """the subject's own anatomy folder only (never `convert subject`, which lists every subject of the protocol): its surfaces one by one, then its images"""
    root = Path(REAL["NXR_IO_BST_PROTOCOL"])
    anat = root / "anat" / REAL["NXR_IO_BST_SUBJECT"]
    files = [str(anat / n) for n in sorted(os.listdir(anat)) if n.startswith("tess_") and n.endswith(".mat") and not n.startswith("._")]
    a, b = real_pair("anatomy")
    pa = run_nxr("convert", "surface", "--dataset", a, "--subject", "S01", "--file", *files, "--bst-root", root, start=1000)
    pb = run_io("convert", "surface", "--dataset", b, "--subject", "S01", "--file", *files, "--bst-root", root, start=1000)
    assert pa.returncode == pb.returncode == 0, (pa.stderr[-800:], pb.stdout[-800:], pb.stderr[-800:])
    assert lines(pa) == lines(pb)
    pa = run_nxr("convert", "mri", "--dataset", a, "--subject", "S01", "--anat", anat, start=5000)
    pb = run_io("convert", "mri", "--dataset", b, "--subject", "S01", "--anat", anat, start=5000)
    assert pa.returncode == pb.returncode == 0, (pa.stdout[-800:], pb.stdout[-800:])
    assert lines(pa) == lines(pb)
    assert_same_dataset(a, b)
