"""The `bst` reading layer against files scipy wrote (a different writer from matio's): every class, complex, logical, unicode text, struct arrays, cells, empties,
sparse; compressed (v7) and uncompressed (v6) and the old v4. Synthetic. Needs nxr's environment (scipy) and the built `nxr-neuro-bst-dump`.

    NXR=<nxr> NXR_NEURO_BST_DUMP=<build>/io/nxr-neuro-bst-dump uv run --project $NXR python -m pytest -q test/parity/test_bst_reading.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.io import savemat

DUMP = Path(os.environ["NXR_NEURO_BST_DUMP"]) if "NXR_NEURO_BST_DUMP" in os.environ else None
pytestmark = pytest.mark.skipif(DUMP is None, reason="set NXR_NEURO_BST_DUMP")


def dump(path) -> dict:
    p = subprocess.run([str(DUMP), str(path)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout)


def fixture() -> dict:
    big = np.array([2 ** 62 + 1, -(2 ** 62) - 1], np.int64)
    return {
        "A": np.arange(1, 7, dtype=float).reshape(3, 2),
        "cube": np.arange(12, dtype=float).reshape(2, 3, 2),
        "i8": np.array([[-5, 7]], np.int8), "u8": np.array([[200, 3]], np.uint8), "i16": np.array([[-300, 300]], np.int16),
        "u16": np.array([[65535, 1]], np.uint16), "i32": np.array([[-70000, 70000]], np.int32), "u32": np.array([[4294967295, 1]], np.uint32),
        "i64": big.reshape(1, 2), "u64": np.array([[2 ** 64 - 1, 1]], np.uint64), "f32": np.array([[1.5, -2.25]], np.float32),
        "Z": np.array([[1 + -1j, 2 + 0.5j]]), "Z32": np.array([[1 + 2j]], np.complex64),
        "L": np.array([[True, False, True]]),
        "S": "Cortex L", "U": "région été 中文", "E": "", "M": np.array(["ab", "cd"]),
        "N": np.zeros((0, 0)),
        "St": np.array([("first", 10.0, {"X": 3.0}), ("second", 20.0, {"X": 4.0})], dtype=[("Name", "O"), ("V", "O"), ("Sub", "O")]),
        "One": {"a": 1.0, "b": "txt", "c": np.array([1.0, 2.0, 3.0])},
        "C": np.array(["MEG", np.array([1.0, 2.0, 3.0])], dtype=object),
        "Sp": sp.csc_matrix(np.array([[0, 2.0, 0], [3.0, 0, 4.0]])),
    }


def expected() -> dict:
    d = lambda cls, dims, **k: {"class": cls, "dims": dims, **k}
    return {
        "A": d("double", [3, 2], data=list(fixture()["A"].ravel(order="F"))),            # column-major, as stored
        "cube": d("double", [2, 3, 2], data=list(fixture()["cube"].ravel(order="F"))),
        "i8": d("int8", [1, 2], data=[-5, 7]), "u8": d("uint8", [1, 2], data=[200, 3]), "i16": d("int16", [1, 2], data=[-300, 300]),
        "u16": d("uint16", [1, 2], data=[65535, 1]), "i32": d("int32", [1, 2], data=[-70000, 70000]), "u32": d("uint32", [1, 2], data=[4294967295, 1]),
        "i64": d("int64", [1, 2], data=[2 ** 62 + 1, -(2 ** 62) - 1]), "u64": d("uint64", [1, 2], data=[2 ** 64 - 1, 1]),
        "f32": d("single", [1, 2], data=[1.5, -2.25]),
        "Z": d("double", [1, 2], complex=True, data=[1, 2], imag=[-1, 0.5]), "Z32": d("single", [1, 1], complex=True, data=[1], imag=[2]),
        "L": d("uint8", [1, 3], logical=True, data=[1, 0, 1]),
        "S": d("char", [1, 8], text=["Cortex L"]), "U": d("char", [1, len(fixture()["U"])], text=[fixture()["U"]]), "E": d("char", [0, 0], text=[]),
        "M": d("char", [2, 2], text=["ab", "cd"]),
        "N": d("double", [0, 0], data=[]),
        "St": d("struct", [1, 2], fields=["Name", "V", "Sub"], elements=[
            {"Name": d("char", [1, 5], text=["first"]), "V": d("double", [1, 1], data=[10]),
             "Sub": d("struct", [1, 1], fields=["X"], elements=[{"X": d("double", [1, 1], data=[3])}])},
            {"Name": d("char", [1, 6], text=["second"]), "V": d("double", [1, 1], data=[20]),
             "Sub": d("struct", [1, 1], fields=["X"], elements=[{"X": d("double", [1, 1], data=[4])}])}]),
        "One": d("struct", [1, 1], fields=["a", "b", "c"], elements=[{"a": d("double", [1, 1], data=[1]), "b": d("char", [1, 3], text=["txt"]),
                                                                     "c": d("double", [1, 3], data=[1, 2, 3])}]),
        "C": d("cell", [1, 2], cells=[d("char", [1, 3], text=["MEG"]), d("double", [1, 3], data=[1, 2, 3])]),
        "Sp": d("sparse", [2, 3], data=[0, 3, 2, 0, 0, 4]),                            # column-major dense image of [[0 2 0],[3 0 4]]
    }


def normalise(x):
    """numbers compare as numbers (an int and its float), the rest as they are"""
    if isinstance(x, dict):
        return {k: normalise(v) for k, v in x.items()}
    if isinstance(x, list):
        return [normalise(v) for v in x]
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and abs(x) < 2 ** 53 else x


@pytest.mark.parametrize("compress", [True, False], ids=["v7_compressed", "v6_uncompressed"])
def test_every_class_reads_as_scipy_wrote_it(compress, tmp_path):
    p = tmp_path / "all.mat"
    savemat(str(p), fixture(), do_compression=compress)
    got, want = dump(p), expected()
    assert sorted(got) == sorted(want)
    for k in want:
        # scipy writes a 1-D array as a row; a "M" char array of two strings is a 2x2 char matrix (rows ab, cd)
        assert normalise(got[k]) == normalise(want[k]), k


def test_a_v4_file_reads(tmp_path):
    p = tmp_path / "old.mat"
    savemat(str(p), {"A": np.arange(1, 7, dtype=float).reshape(3, 2), "B": np.array([[1.5]])}, format="4")
    got = dump(p)
    assert got["A"]["data"] == [1, 3, 5, 2, 4, 6] and got["A"]["dims"] == [3, 2] and got["B"]["data"] == [1.5]


def test_an_empty_struct_array_and_a_struct_without_fields(tmp_path):
    p = tmp_path / "e.mat"
    savemat(str(p), {"S0": np.array([], dtype=[("a", "O")]), "Cell0": np.empty((0,), dtype=object)})
    got = dump(p)
    assert got["S0"]["class"] == "struct" and got["S0"]["elements"] == [] and got["Cell0"]["class"] == "cell" and got["Cell0"]["cells"] == []
