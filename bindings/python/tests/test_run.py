"""The pipeline level: `nn.run`, `nn.list` and `nn.read` on a synthetic dataset made in the test (the library's `synthetic_dataset`: a recording of known signals, one bad span).

`run("psd", ...)` stores what `nxr-neuro run psd` stores (the C++ test `test_face` runs both on one dataset and compares them); here the stored PSD is checked against the pure
function `nn.compute.signal.psd` on the stored samples, with the stored recording's bad span left out as the pipeline leaves it out."""

import numpy as np
import pytest

import nxr_neuro as nn

SFREQ, SECONDS, CHANNELS = 100.0, 40.0, 3


@pytest.fixture
def dataset(tmp_path):
    return nn._core._synthetic_dataset(str(tmp_path), "demo", "S1", "rest", CHANNELS, SFREQ, SECONDS)


def _good(n):
    t = np.arange(n) / SFREQ
    return ~((t >= 0.3 * SECONDS) & (t < 0.375 * SECONDS))


def _read(dataset, cls, name, **kw):
    return getattr(nn.read, cls)(dataset=dataset, subject="S1", name=name, **kw)


def test_run_needs_the_time_range(dataset):
    for kwargs in ({}, {"from_": 0.0}, {"to": 10.0}, {"from_": 0.0, "to": 10.0, "all": True}):
        with pytest.raises(nn.InvalidArgumentError):
            nn.run("psd", dataset, "S1", recording="rest", **kwargs)
    with pytest.raises(nn.InvalidArgumentError, match="no parameter 'bogus'"):
        nn.run("psd", dataset, "S1", recording="rest", all=True, bogus=1)


def test_psd_is_the_pure_functions_result(dataset):
    lines = []
    r = nn.run("psd", dataset, "S1", recording="rest", preset="scipy", segment_seconds=2.0, all=True, progress=lines.append)
    assert r["cache"] == "computed" and lines and all(isinstance(line, dict) for line in lines)
    assert [s["name"] for s in r["stored"]] == ["rest_psd_welch"]
    got = _read(dataset, "statistic", "rest_psd_welch")
    assert got["class"] == "statistic" and got["measure"] == "power_density"
    assert got.arrays["values"].shape == (CHANNELS, 101) and got.arrays["values"].dtype == np.float32       # the stored precision (the recording's)
    x = _read(dataset, "field", "rest", all=True).arrays["values"]
    assert x.shape == (CHANNELS, int(SECONDS * SFREQ)) and x.dtype == np.float32
    want = nn.compute.signal.psd(x, SFREQ, segment_samples=200, good=_good(x.shape[1]))     # scipy's set: hann (periodic), constant detrend, 50 % overlap, density
    assert 0 < want.segments_skipped < want.segments + want.segments_skipped
    np.testing.assert_allclose(got.arrays["values"], want.psd, rtol=1e-6)
    # without the bad span left out: the plain PSD
    nn.run("psd", dataset, "S1", recording="rest", preset="scipy", segment_seconds=2.0, all=True, name="plain", bad_segments="ignore")
    np.testing.assert_allclose(_read(dataset, "statistic", "plain").arrays["values"], nn.compute.signal.psd(x, SFREQ, segment_samples=200).psd, rtol=1e-6)


def test_lists(dataset, tmp_path):
    names = {p.name: p for p in nn.list.pipelines()}
    assert "psd" in names and "global_bursts" in names and names["psd"].form == "definition" and names["global_bursts"].form == "definition"
    assert all(p["meaning"] for p in names["psd"].parameters) and "segment_seconds" in names["psd"].help()
    assert [d["name"] for d in nn.list.datasets(datastore=tmp_path)["datasets"]] == ["demo"]
    assert [s["name"] for s in nn.list.subjects(dataset=dataset)["subjects"]] == ["S1"]
    with pytest.raises(nn.InvalidArgumentError):
        nn.list.subjects(dataset=dataset, where="age > 3")
    nn.run("psd", dataset, "S1", recording="rest", all=True)
    assert [r["name"] for r in nn.list.results(dataset=dataset, subject="S1")["results"]] == ["rest_psd_welch"]
    assert [r["name"] for r in nn.list.recordings(dataset=dataset, subject="S1")["recordings"]] == ["rest"]


def test_read_a_recording_by_range_and_rows(dataset):
    whole = _read(dataset, "field", "rest", all=True)
    assert whole["rate"] == SFREQ and whole["origin"] == 0.0 and whole["channels"] == "sensors"
    part = _read(dataset, "field", "rest", from_=10.0, to=25.0, rows=[2, 0])   # stored in chunks of 700 samples: read across two chunk edges
    assert part.arrays["values"].shape == (2, 1500) and part.arrays["values"].dtype == np.float32
    np.testing.assert_array_equal(part.arrays["values"], whole.arrays["values"][[2, 0], 1000:2500])
    shape = _read(dataset, "field", "rest", shape_only=True)
    assert shape.arrays == {} and shape["n_samples"] == int(SECONDS * SFREQ) and shape["n_channels"] == CHANNELS
    for kwargs in ({"from_": 0.0}, {"rows": [CHANNELS], "all": True}, {"projectors": "bogus", "all": True}):
        with pytest.raises(nn.InvalidArgumentError):
            _read(dataset, "field", "rest", **kwargs)
