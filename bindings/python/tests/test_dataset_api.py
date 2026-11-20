"""The dataset API (docs/specs/2026-11-19-io-service.md): the command line's commands as functions on synthetic datasets made in the test. Datasets, subjects and packages
are created, listed and shown; every class is added from NumPy arrays and read back byte for byte in the input's precision; a pipeline is compiled; the writer lock of a
handle refuses another process's write until it is closed."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import nxr_neuro as nn

SFREQ, SECONDS, CHANNELS = 100.0, 40.0, 3


@pytest.fixture
def demo(tmp_path):
    """A one-database dataset with a recording of known signals (`rest`, on the channel manifold `sensors`) and one bad span."""
    return nn._core._synthetic_dataset(str(tmp_path), "demo", "S1", "rest", CHANNELS, SFREQ, SECONDS)


def test_datasets_subjects_and_packages(tmp_path):
    ds = tmp_path / "study"
    made = nn.create.dataset(dataset=ds)
    assert isinstance(made, nn.Result) and made.arrays == {}
    nn.create.subject(dataset=ds, name="S1")
    nn.create.subject(dataset=ds, name="S2")
    assert [s["name"] for s in nn.list.subjects(dataset=ds)["subjects"]] == ["S1", "S2"]
    assert [s["name"] for s in nn.list.subjects(dataset=ds, subjects=["S2"])["subjects"]] == ["S2"]
    assert nn.show.subject(dataset=ds, subject="S1")
    assert nn.show.dataset(dataset=ds)
    assert [m["name"] for m in nn.list.members(dataset=ds)["subjects"]] == ["S1", "S2"]
    with pytest.raises(nn.AlreadyExists):
        nn.create.subject(dataset=ds, name="S1")
    with pytest.raises(TypeError, match="name"):
        nn.create.subject(dataset=ds)   # a required option is required
    # a package of its own, with a synthetic recording (progress lines as dicts)
    pkg = tmp_path / "P1.nxr.zarr"
    p = nn.create.package(package=pkg, name="P1", home_dataset="study")
    assert p["subject_id"]
    lines = []
    nn.create.recording(package=pkg, synthetic=True, name="rest", channels=2, sfreq=200.0, duration=4.0, frequency=[10.0, 12.0], seed=3, progress=lines.append)
    assert all(isinstance(line, dict) for line in lines)
    assert nn.show.package(package=pkg)["record"]["name"] == "P1"
    assert nn.verify.package(package=pkg)["ok"] is True
    with pytest.raises(nn.InvalidArgumentError, match="synthetic"):
        nn.create.recording(package=pkg, name="other")   # say what to make, as the command line refuses it


def test_read_subject_and_list_each_class(demo):
    """`read subject` describes every node, no arrays; `list <class>` pages the same descriptions; neither gives a column of the schema."""
    sub = nn.read.subject(dataset=demo, subject="S1")
    assert sub["schema"] == "nxr-neuro-subject@1" and sub["subject"]["name"] == "S1"
    rest = next(f for f in sub["fields"] if f["name"] == "rest")
    assert rest["type"]["class"] == "field" and rest["type"]["sfreq"] == SFREQ and sub.arrays == {}
    assert "value_type" not in rest and "cells" not in rest
    assert set(nn.read.subject(dataset=demo, subject="S1", classes=["fields"])) >= {"schema", "subject", "fields"}
    assert "manifolds" not in nn.read.subject(dataset=demo, subject="S1", classes="fields")
    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)
    assert not [k for k in keys(sub) if k.endswith("_id") or k.endswith("_json")]
    for cls in ("manifolds", "fields", "selections", "operators", "statistics", "dictionaries", "spaces", "acquisitions", "jobs"):
        rows = getattr(nn.list, cls)(dataset=demo, subject="S1")["rows"]
        if cls != "jobs":
            assert [{k: v for k, v in r.items() if k != "subject"} for r in rows] == sub[cls]
    first = nn.list.manifolds(dataset=demo, subject="S1", limit=1)
    assert len(first["rows"]) == 1 and first["next"]
    rest_of = nn.list.manifolds(dataset=demo, subject="S1", after=first["next"])
    assert [r["name"] for r in first["rows"] + rest_of["rows"]] == [m["name"] for m in sub["manifolds"]]
    assert nn.list.fields(dataset=demo, subject="S1", name="res", columns="name,kind")["rows"] == [{"name": "rest", "kind": rest["kind"]}]
    assert nn.show.dataset(dataset=demo)["nodes"]["fields"] == len(sub["fields"])
    with pytest.raises(nn.InvalidArgumentError, match="is not a column of fields"):
        nn.list.fields(dataset=demo, columns=["geometry_id"])


def test_read_each_class(demo):
    with pytest.raises(nn.InvalidArgumentError, match="a time range is required"):
        nn.read.field(dataset=demo, subject="S1", name="rest")
    f = nn.read.field(dataset=demo, subject="S1", name="rest", from_=0.0, to=1.0)
    assert f["class"] == "field" and f.arrays["values"].shape == (CHANNELS, 100) and f.arrays["values"].dtype == np.float32
    whole = nn.read.field(dataset=demo, subject="S1", name="rest", all=True, rows=[2, 0])
    assert whole.arrays["values"].shape == (2, int(SECONDS * SFREQ))
    np.testing.assert_array_equal(whole.arrays["values"][:, :100], f.arrays["values"][[2, 0]])
    assert nn.read.field(dataset=demo, subject="S1", name="rest", shape_only=True).arrays == {}
    m = nn.read.manifold(dataset=demo, subject="S1", name="sensors")
    assert m.arrays["positions"].shape == (CHANNELS, 3) and m["names"] == ["C0", "C1", "C2"]
    q = nn.read.manifold(dataset=demo, subject="S1", name="sensors", family="quality")
    names = {e["code"]: e["name"] for e in q["dictionary"]}
    assert [names[c] for c in q.arrays["codes"]] == ["good", "good", "bad"]
    b = nn.read.selection(dataset=demo, subject="S1", name="rest_bad_segments")
    assert b.arrays["start"].tolist() == [0.3 * SECONDS] and b.arrays["stop"].tolist() == [0.375 * SECONDS]
    with pytest.raises(nn.NoSuchRow):
        nn.read.statistic(dataset=demo, subject="S1", name="nothing")
    # read dataset: every array the rows name; one array by subject and path
    listing = nn.read.dataset(demo)
    assert listing["subjects"] and listing["arrays"]
    one = listing["arrays"][0]
    got = nn.read.dataset(demo, array=(one["subject"], one["path"]))
    arr = next(iter(got.arrays.values()))
    assert list(arr.shape) == one["shape"] and arr.dtype == np.dtype(one["dtype"]) and got["checksum"] == one["checksum"]


def test_add_each_class_and_read_it_back(demo, tmp_path):
    def read(cls, name, **kw):
        return getattr(nn.read, cls)(dataset=demo, subject="S1", name=name, **kw)

    def add(cls, **kw):
        return getattr(nn.add, cls)(dataset=demo, subject="S1", **kw)

    w = np.array([0.5, -1.25, 3.0], dtype=np.float32)
    a = add("field", name="w", manifold="sensors", values=w, unit="mm", description="a test map")
    assert a["class"] == "field"
    back = read("field", "w")
    assert back.arrays["values"].dtype == np.float32 and back.arrays["values"].tobytes() == w.tobytes()   # precision follows the input
    assert "values.npy" in json.dumps(back["producer"])                                               # the input's file name, never a path
    with pytest.raises(nn.AlreadyExists):
        add("field", name="w", manifold="sensors", values=w)
    assert add("field", name="w", manifold="sensors", values=w, replace=True)["replaced"]               # moved aside, never deleted
    with pytest.raises(nn.InvalidArgumentError, match=r"is \[4\]"):
        add("field", name="w4", manifold="sensors", values=np.arange(4.0))

    add("line", name="epoch_time", n=50, spacing=0.01, unit="s")
    add("line", name="freq", n=4, spacing=1.0, origin=1.0, unit="Hz", dimension="frequency")
    series = 0.25 * np.arange(CHANNELS * 50, dtype=np.float64).reshape(CHANNELS, 50)
    add("field", name="series", manifold="sensors", time="epoch_time", values=series)
    np.testing.assert_array_equal(read("field", "series", all=True).arrays["values"], series)

    parts = np.array([0, 1, 0], dtype=np.int32)
    assert add("selection", name="parts", manifold="sensors", type="partition", members=parts, entries=["left", "right"])["members"] == 2
    np.testing.assert_array_equal(read("selection", "parts").arrays["values"], parts)
    add("selection", name="marks", manifold="epoch_time", type="spans", spans=np.array([[0.05, 0.10, 0], [0.2, 0.35, 1]]))
    s = read("selection", "marks")
    assert s.arrays["start"].tolist() == [0.05, 0.2] and s.arrays["code"].tolist() == [0, 1]
    add("selection", name="pick", manifold="sensors", type="indices", indices=np.array([2, 0], dtype=np.int32))

    add("measurements", selection="parts", field="w", measure={"mean": np.array([1.5, -2.0]), "max": np.array([3.0, -1.25])})
    np.testing.assert_array_equal(read("selection", "parts", measure="mean", field="w").arrays["value"], [1.5, -2.0])
    q = nn.query.measurements(dataset=demo, subject="S1", selection="parts", measure="max", field="w", where="io")   # a stored ingredient, read as stored
    assert [row["value"] for row in q["result"]["rows"]] == [3.0, -1.25]
    q = nn.query.measurements(dataset=demo, subject="S1", selection="parts", measure="max", field="w")              # the default, in memory: the same values
    assert q["result"]["values"]["data"] == [3.0, -1.25]

    v = (0.5 * np.arange(12)).astype(np.float32).reshape(CHANNELS, 4)
    add("statistic", name="psd", measure="power_density", method="welch", axes=["channel:sensors", "frequency:freq"], of_field="rest", values=v)
    st = read("statistic", "psd")
    assert st.arrays["values"].dtype == np.float32 and st.arrays["values"].tobytes() == v.tobytes()

    dense = np.array([[1, 0, 0], [0, 2, 0], [0.5, 0, 3]])
    add("operator", name="mix", from_manifold="sensors", to_manifold="sensors", values=dense)
    np.testing.assert_array_equal(read("operator", "mix").arrays["values"], dense)
    add("operator", name="sparse", from_manifold="sensors", to_manifold="sensors", data=np.array([4.0, 5.0]), indices=np.array([0, 2], dtype=np.int32),
        indptr=np.array([0, 1, 1, 2], dtype=np.int32))
    sp = read("operator", "sparse", out=tmp_path / "sparse.npy")   # `out` writes the .npy files as the command does, and says where
    assert sp.arrays["data"].tolist() == [4.0, 5.0] and sp.arrays["indices"].tolist() == [0, 2]
    assert set(sp["files"]) == {"data", "indices", "indptr"} and all(os.path.exists(p) for p in sp["files"].values())

    pos = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0.5, 0.5, 0.5]])
    add("points", name="electrodes", positions=pos)
    np.testing.assert_array_equal(read("manifold", "electrodes").arrays["positions"], pos)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    assert add("surface", name="tetra", vertices=pos, faces=faces, no_primary=True)["n_vertices"] == 4
    assert read("manifold", "tetra").arrays["cells_2"].shape == (4, 3)
    names = {r["name"] for r in nn.list.results(dataset=demo, subject="S1")["results"]}
    assert {"w", "psd"} <= names


def test_compile_and_run_with_from_and_to(demo):
    c = nn.compile("psd", dataset=demo, subject="S1", set={"recording": "rest"})
    assert c["types"] == "stored" and c["plans"]
    r = nn.run("psd", demo, "S1", recording="rest", segment_seconds=2.0, from_=0.0, to=20.0)
    assert r["cache"] == "computed" and [s["name"] for s in r["stored"]] == ["rest_psd_welch"]
    assert nn.run("psd", demo, "S1", recording="rest", segment_seconds=2.0, from_=0.0, to=20.0)["cache"] == "hit"
    psd = nn.read.statistic(dataset=demo, subject="S1", name="rest_psd_welch")
    assert psd.arrays["values"].shape[0] == CHANNELS and psd["span"]["stop_sample"] == 2048
    with pytest.raises(nn.InvalidArgumentError, match="time range"):
        nn.run("psd", demo, "S1", recording="rest")
    dry = nn.run("psd", demo, "S1", recording="rest", all=True, dry_run=True)
    assert all(run["cache"] == "planned" for run in dry["runs"])


def test_a_write_handle_holds_the_lock_across_processes(demo):
    child = textwrap.dedent(f"""
        import sys, nxr_neuro as nn
        try:
            nn.add.line(dataset={demo!r}, subject="S1", name=sys.argv[1], n=3, spacing=1.0, unit="s")
        except nn.WriterLockHeld:
            sys.exit(3)
    """)

    def other_process(name):
        return subprocess.run([sys.executable, "-c", child, name], capture_output=True, text=True).returncode

    with nn.open(demo, mode="write") as d:
        assert other_process("blocked") == 3                                   # refused while the handle holds the lock
        d.add.line(subject="S1", name="mine", n=3, spacing=1.0, unit="s")     # the handle's own commands borrow it
        assert d.read.manifold(subject="S1", name="mine")["class"] == "manifold"
        assert d.show.dataset()
    assert d.closed
    assert other_process("after") == 0                                         # released by close()
    with pytest.raises(nn.InvalidArgumentError, match="closed"):
        d.show.dataset()
    r = nn.open(demo)                                                          # a read handle takes no lock
    assert r.read.field(subject="S1", name="rest", from_=0.0, to=1.0).arrays["values"].shape == (CHANNELS, 100)
    assert other_process("while_read") == 0


def test_a_read_handle_keeps_its_database_open_and_sees_what_is_stored_meanwhile(demo):
    with nn.open(demo) as d:
        assert d.read.manifold(subject="S1", name="sensors")["class"] == "manifold"
        assert d.list.recordings(subject="S1")
        nn.add.line(dataset=demo, subject="S1", name="stored_between", n=4, spacing=1.0, unit="s")   # another call, no handle, while the handle's reader is open
        assert d.read.manifold(subject="S1", name="stored_between")["class"] == "manifold"
    with pytest.raises(nn.InvalidArgumentError, match="closed"):
        d.read.manifold(subject="S1", name="sensors")
