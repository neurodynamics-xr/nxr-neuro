"""Parity of three Brainstorm-import fixes (nxr v1.8.0): the duplicate-kernel check, an unreadable subject image, and source_orient. SYNTHETIC files only.

    NXR=<path to nxr> NXR_NEURO=<build>/cli/nxr-neuro uv run --project $NXR python -m pytest -q test/parity/test_bst_import_fixes_parity.py

Both tools run the same command into two copies of one starting dataset; the output lines (ids and paths aside) and every row, zarr.json and chunk are equal.
"""
from __future__ import annotations

import json
import os
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
    from test_sensors_mri_parity import build_images, write_image  # noqa: E402

RUN1, RUN2 = "sub-S01_task-rest_run-01", "sub-S01_task-rest_run-02"
KERNEL = "results_MN_EEG_KERNEL_000000_0000.mat"


def _cell(*items):
    c = np.empty((1, len(items)), dtype=object)
    for i, x in enumerate(items):
        c[0, i] = x
    return c


def _load(path):
    return {k: v for k, v in loadmat(str(path), simplify_cells=True).items() if not k.startswith("__")}


def _strip(L):
    return [{k: v for k, v in x.items() if k not in ("store", "id")} for x in L]


def _both(tmp_path, *args, **kw):
    a, b = starting_pair(tmp_path, subjects=())
    return a, b, run_nxr(*args[0], str(a), *args[1], start=1000), run_io(*args[0], str(b), *args[1], start=1000)


def test_a_kernel_of_the_same_name_is_refused_in_its_acquisition_and_allowed_in_another(tmp_path):
    root = tmp_path / "proto"
    cf.build_protocol(root)
    a, b = starting_pair(tmp_path, subjects=())
    for run, ds in ((run_nxr, a), (run_io, b)):
        assert run("convert", "subject", root, "--subject", "S01", "--dataset", ds, start=1000).returncode == 0
    src = root / "data" / "S01" / f"@raw{RUN1}"
    twin = tmp_path / "twin"
    twin.mkdir()
    k = _load(src / KERNEL)
    k["ImagingKernel"] = np.ones_like(k["ImagingKernel"])          # other content under the same name: only the acquisition can refuse it
    savemat(str(twin / KERNEL), k)
    for f in src.glob("channel*.mat"):
        (twin / f.name).write_bytes(f.read_bytes())

    def add(ds, run, acquisition, start):
        return run("convert", "inverse", "--dataset", ds, "--subject", "S01", "--file", twin / KERNEL, "--acquisition", acquisition, "--bst-root", root, start=start)

    pa, pb = add(a, run_nxr, RUN1, 2000), add(b, run_io, RUN1, 2000)
    assert pa.returncode == pb.returncode == 1, (pa.stdout, pb.stdout)
    for p in (pa, pb):
        assert "is already in acquisition " + RUN1 in lines(p)[-1]["message"] and "refusing to add it twice" in lines(p)[-1]["message"]
    assert lines(pa)[-1] == lines(pb)[-1]
    pa, pb = add(a, run_nxr, RUN2, 3000), add(b, run_io, RUN2, 3000)
    assert pa.returncode == pb.returncode == 0, (pa.stdout + pa.stderr, pb.stdout + pb.stderr)
    assert _strip(lines(pa)) == _strip(lines(pb))
    assert_same_dataset(a, b)


def test_an_unreadable_subject_image_is_skipped_alike_and_the_rest_imports(tmp_path):
    anat = tmp_path / "anat" / "S01"
    build_images(anat)
    good = (anat / "subjectimage_MRI_T1.mat").read_bytes()
    (anat / "subjectimage_zz_bad.mat").write_bytes(good[: len(good) // 2])
    a, b = starting_pair(tmp_path)
    pa = run_nxr("convert", "mri", "--dataset", a, "--subject", "S01", "--anat", anat, start=1000)
    pb = run_io("convert", "mri", "--dataset", b, "--subject", "S01", "--anat", anat, start=1000)
    assert pa.returncode == pb.returncode == 0, pa.stdout + pa.stderr + pb.stdout + pb.stderr
    skipped = [x for x in lines(pb) if x["stage"] == "skipped"]
    assert skipped == [x for x in lines(pa) if x["stage"] == "skipped"]
    assert skipped == [{"stage": "skipped", "file": "subjectimage_zz_bad.mat", "reason": "subjectimage_zz_bad.mat: not a readable MATLAB file"}]
    assert _strip(lines(pa)) == _strip(lines(pb)) and any(x["stage"] == "mri" for x in lines(pb))
    assert_same_dataset(a, b)


@pytest.mark.parametrize("orient, want", [
    ("free", ["free"]),                            # a char
    (_cell("free"), ["free"]),                     # a 1x1 cell
    (_cell("free", "loose"), ["free", "loose"]),   # a cell of two
    (np.zeros((0, 0)), []),                        # empty
], ids=["char", "cell-1x1", "cell-1x2", "empty"])
def test_source_orient_is_a_list_of_strings_in_both(tmp_path, orient, want):
    root = tmp_path / "proto"
    cf.build_protocol(root)
    for f in (root / "data").rglob("results_*KERNEL*.mat"):
        k = _load(f)
        k["Options"] = {"InverseMethod": "minnorm", "InverseMeasure": "amplitude", "SourceOrient": orient}
        savemat(str(f), k)
    a, b = starting_pair(tmp_path, subjects=())
    pa = run_nxr("convert", "subject", root, "--subject", "S01", "--dataset", a, start=1000)
    pb = run_io("convert", "subject", root, "--subject", "S01", "--dataset", b, start=1000)
    assert pa.returncode == pb.returncode == 0, (pa.stdout[-600:], pb.stdout[-600:])
    assert_same_dataset(a, b)
    from nxr.io.database import Database  # noqa: F401  (the rows are compared above; this reads one)
    import sqlite3
    con = sqlite3.connect(str(b / "dataset.sqlite"))
    pj = [json.loads(r[0]) for r in con.execute("SELECT producer_json FROM operator WHERE kind = 'inverse kernel'")]
    assert pj and all(p["source_orient"] == want for p in pj), pj
