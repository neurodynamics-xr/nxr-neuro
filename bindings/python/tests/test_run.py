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


def test_run_needs_the_time_range(dataset):
    for kwargs in ({}, {"start": 0.0}, {"stop": 10.0}, {"start": 0.0, "stop": 10.0, "all": True}):
        with pytest.raises(nn.InvalidArgumentError):
            nn.run("psd", dataset, "S1", recording="rest", **kwargs)
    with pytest.raises(nn.InvalidArgumentError, match="no parameter 'bogus'"):
        nn.run("psd", dataset, "S1", recording="rest", all=True, bogus=1)


def test_psd_is_the_pure_functions_result(dataset):
    lines = []
    r = nn.run("psd", dataset, "S1", recording="rest", preset="scipy", segment_seconds=2.0, all=True, progress=lines.append)
    assert r.cache == "computed" and lines and all(isinstance(line, dict) for line in lines)
    assert [s["name"] for s in r.stored] == ["rest_psd_welch"]
    got = r.read()
    assert got.meta["class"] == "statistic" and got.meta["measure"] == "power_density"
    assert got.values.shape == (CHANNELS, 101) and got.values.dtype == np.float64
    x = nn.read(dataset, "S1", "rest").values
    assert x.shape == (CHANNELS, int(SECONDS * SFREQ)) and x.dtype == np.float32
    want = nn.compute.signal.psd(x, SFREQ, segment_samples=200, good=_good(x.shape[1]))     # scipy's set: hann (periodic), constant detrend, 50 % overlap, density
    assert 0 < want.segments_skipped < want.segments + want.segments_skipped
    np.testing.assert_allclose(got.values, want.psd, rtol=1e-6)
    # without the bad span left out: the plain PSD
    r2 = nn.run("psd", dataset, "S1", recording="rest", preset="scipy", segment_seconds=2.0, all=True, name="plain", bad_segments="ignore")
    np.testing.assert_allclose(r2.read().values, nn.compute.signal.psd(x, SFREQ, segment_samples=200).psd, rtol=1e-6)


def test_a_rerun_is_a_cache_hit_and_a_range_is_a_range(dataset):
    a = nn.run("psd", dataset, "S1", recording="rest", segment_seconds=2.0, start=0.0, stop=20.0)
    assert a.cache == "computed"
    assert nn.run("psd", dataset, "S1", recording="rest", segment_seconds=2.0, start=0.0, stop=20.0).cache == "hit"
    b = nn.run("psd", dataset, "S1", recording="rest", segment_seconds=2.0, start=0.0, stop=20.0, name="again")
    np.testing.assert_array_equal(a.read().values, b.read().values)
    assert a.read().meta["span"]["stop_sample"] == 2048   # 20 s = 2000 samples, snapped to the 2^k-sample tile boundary of the default set


def test_lists(dataset, tmp_path):
    names = {p.name: p for p in nn.list.pipelines()}
    assert "psd" in names and "global_bursts" in names and names["psd"].form == "definition" and names["global_bursts"].form == "definition"
    assert all(p["meaning"] for p in names["psd"].parameters) and "segment_seconds" in names["psd"].help()
    assert [d["name"] for d in nn.list.datasets(tmp_path)] == ["demo"]
    assert [s["name"] for s in nn.list.subjects(dataset)] == ["S1"]
    with pytest.raises(nn.InvalidArgumentError):
        nn.list.subjects(dataset, where="age > 3")
    nn.run("psd", dataset, "S1", recording="rest", all=True)
    assert [r["name"] for r in nn.list.results(dataset, "S1")] == ["rest_psd_welch"]
    assert [r["name"] for r in nn.list.results(dataset, "S1", "recordings")] == ["rest"]


def test_read_refuses_what_is_not_stored(dataset):
    with pytest.raises(nn.Error) as e:
        nn.read(dataset, "S1", "nothing_of_that_name")
    assert e.value.code == "no_such_row"


def test_read_a_recording_by_span_and_rows(dataset):
    whole = nn.read(dataset, "S1", "rest")
    assert whole.meta["rate"] == SFREQ and whole.meta["origin"] == 0.0 and whole.meta["channels"] == "sensors"
    part = nn.read(dataset, "S1", "rest", start=1000, stop=2500, rows=[2, 0])   # stored in chunks of 700 samples: read across two chunk edges
    assert part.values.shape == (2, 1500) and part.values.dtype == np.float32
    np.testing.assert_array_equal(part.values, whole.values[[2, 0], 1000:2500])
    assert part.meta["span"]["start_sample"] == 1000 and part.meta["rows"] == [2, 0]
    shape = nn.read(dataset, "S1", "rest", reading="shape")
    assert shape.arrays == {} and shape.meta["n_samples"] == int(SECONDS * SFREQ) and shape.meta["n_channels"] == CHANNELS
    for kwargs in ({"start": 0}, {"start": 0, "stop": 10**6}, {"rows": [CHANNELS]}, {"reading": "bogus"}, {"projectors": "bogus"}):
        with pytest.raises(nn.InvalidArgumentError):
            nn.read(dataset, "S1", "rest", **kwargs)


def test_read_a_manifold_its_labelling_and_spans(dataset):
    m = nn.read(dataset, "S1", "sensors")
    assert m.meta["class"] == "manifold" and m.positions.shape == (CHANNELS, 3) and m.meta["names"] == ["C0", "C1", "C2"]
    q = nn.read(dataset, "S1", "sensors", family="quality")
    names = {e["code"]: e["name"] for e in q.meta["dictionary"]}
    assert q.codes.dtype == np.int64 and [names[c] for c in q.codes] == ["good", "good", "bad"]
    with pytest.raises(nn.Error) as e:
        nn.read(dataset, "S1", "sensors", family="nothing")
    assert e.value.code == "no_such_row"
    with pytest.raises(nn.InvalidArgumentError):
        nn.read(dataset, "S1", "sensors", start=0, stop=10)
    b = nn.read(dataset, "S1", "rest_bad_segments")
    names = {e["code"]: e["name"] for e in b.meta["dictionary"]}
    assert b.start.tolist() == [0.3 * SECONDS] and b.stop.tolist() == [0.375 * SECONDS] and [names[c] for c in b.code] == ["BAD"]
