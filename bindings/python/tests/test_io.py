"""The readers against the repository's synthetic files and their goldens (MNE-Python, nibabel, pandas)."""

import numpy as np
import pytest

import nxr_neuro as nn

RECORDING = {"edf": nn.io.open_edf, "bdf": nn.io.open_edf, "brainvision": nn.io.open_brainvision, "eeglab": nn.io.open_eeglab, "fif": nn.io.open_fif}


def cases(golden, name, only=None):
    return [(k, v) for k, v in golden(name)["cases"].items() if "data" in v and (only is None or k in only)]


def check_recording(rec, case, want):
    h = rec.header
    n_ch = len(case["mne_channels"])
    n = case.get("n_times", case.get("n_samples"))
    assert [c["name"] for c in h["channels"]] == case["mne_channels"]
    assert h["sampling_rate"] == pytest.approx(case["sfreq"]) and h["n_samples"] == n
    want = want.reshape(n_ch, n)
    x = rec.read_all()
    assert x.shape == (n_ch, n)
    np.testing.assert_allclose(x, want, rtol=2e-6, atol=1e-12)
    # a block read does not depend on the block, nor on the channels asked
    a, b = n // 3, 2 * n // 3
    np.testing.assert_array_equal(rec.read(a, b), x[:, a:b])
    np.testing.assert_array_equal(rec.read(a, b, first_channel=1, last_channel=n_ch), x[1:, a:b])
    # BrainVision's own "New Segment" marker is a marker like any other here; MNE keeps it out of the annotations
    ann = [(round(e["onset"], 6), round(e["duration"], 6), e["label"]) for e in h["annotations"] if not e["label"].startswith("New Segment")]
    assert ann == [(round(e["onset"], 6), round(e["duration"], 6), e["label"]) for e in case["annotations"]]


@pytest.mark.parametrize("fmt", ["edf", "brainvision", "eeglab"])
def test_recordings_equal_the_mne_goldens(golden, f32, formats_dir, fmt):
    for name, case in cases(golden, fmt):
        if case.get("gaps"):
            continue
        rec = RECORDING[case.get("format", fmt) if case.get("format") in ("bdf",) else fmt](str(formats_dir / "data" / case["file"]))
        check_recording(rec, case, f32(case["data"]))


def test_edf_gaps_read_as_zeros(golden, f32, formats_dir):
    case = golden("edf")["cases"]["plus_d"]
    rec = nn.io.open_edf(formats_dir / "data" / case["file"])
    assert rec.header["gaps"] == [(g["first"], g["count"]) for g in case["gaps"]]
    x = rec.read_all()
    g = case["gaps"][0]
    assert not x[:, g["first"]:g["first"] + g["count"]].any()


def test_read_helpers_state_the_span(formats_dir):
    path = formats_dir / "data" / "edf_plain.edf"
    x, header = nn.io.read_edf(path, start=100, stop=200)
    assert x.shape[1] == 100 and header["format"].startswith("edf")
    assert nn.io.read_edf(path, all=True)[0].shape[1] == header["n_samples"]
    with pytest.raises(ValueError, match="span"):
        nn.io.read_edf(path)
    with pytest.raises(ValueError):
        nn.io.read_edf(path, start=0, stop=10, all=True)
    with pytest.raises(nn.InvalidArgumentError):
        nn.io.read_edf(path, start=0, stop=10**9)


def test_refusals_are_errors_with_codes(formats_dir):
    data = formats_dir / "data"
    with pytest.raises(nn.FormatError) as e:
        nn.io.open_edf(data / "edf_bad_version.edf")
    assert e.value.code == "format_error" and isinstance(e.value, nn.Error)
    with pytest.raises(nn.UnsupportedError) as e:
        nn.io.open_edf(data / "edf_mixed_rates.edf")
    assert isinstance(e.value, NotImplementedError)
    with pytest.raises(nn.UnsupportedError):
        nn.io.open_brainvision(data / "bv_ascii.vhdr")
    with pytest.raises(nn.IOError_):
        nn.io.open_edf(data / "no_such_file.edf")
    # a rate named lets a mixed-rate file be read
    named = nn.io.open_edf(data / "edf_mixed_rates.edf", sampling_rate=100.0)
    assert named.header["sampling_rate"] == 100.0


def test_precision_follows_the_file(formats_dir):
    assert nn.io.open_eeglab(formats_dir / "data" / "eeglab_fdt.set").read_all().dtype == np.float32
    assert nn.io.open_edf(formats_dir / "data" / "edf_plain.edf").read_all().dtype == np.float32


def test_nifti_equals_nibabel_goldens(golden, formats_dir):
    for name, case in golden("nifti")["cases"].items():
        if "data" not in case:
            continue
        data, h = nn.io.read_nifti(formats_dir / "data" / case["file"])
        assert list(data.shape) == case["shape"] == list(h["shape"])
        np.testing.assert_allclose(data.ravel(order="F"), case["data"], rtol=1e-6, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(h["affine"][:3].ravel(), case["best_affine"], rtol=1e-6, atol=1e-9, err_msg=name)
        assert h["affine"].shape == (4, 4) and list(h["affine"][3]) == [0, 0, 0, 1]
        assert h["version"] == case["version"] and h["descrip"] == case["descrip"]


def test_nifti_pair_and_refusals(formats_dir):
    data, h = nn.io.read_nifti(formats_dir / "data" / "nifti_pair.hdr")
    assert data.ndim == 3 and h["version"] == 1
    with pytest.raises(nn.UnsupportedError):
        nn.io.read_nifti(formats_dir / "data" / "nifti_complex64.nii")
    with pytest.raises(nn.FormatError):
        nn.io.read_nifti(formats_dir / "data" / "not_nifti.nii")


def test_gifti_equals_nibabel_goldens(golden, formats_dir):
    for name, case in golden("gifti")["cases"].items():
        img = nn.io.read_gifti(formats_dir / "data" / case["file"])
        assert len(img["arrays"]) == len(case["arrays"])
        for got, want in zip(img["arrays"], case["arrays"]):
            assert got["intent"] == want["intent"] and got["datatype"] == want["datatype"]
            assert list(got["data"].shape) == want["dims"]
            np.testing.assert_allclose(got["data"].ravel(), want["data"], rtol=1e-6, atol=1e-9, err_msg=name)
            for k, v in want["metadata"].items():
                assert got["metadata"][k] == v


@pytest.mark.parametrize("name", ["lf", "crlf", "bom"])
def test_tsv_equals_the_pandas_golden(golden, formats_dir, name):
    want = golden("tsv")[name]
    t = nn.io.read_tsv(formats_dir / "data" / f"tsv_{name}.tsv")
    assert list(t) == want["columns"]
    for j, col in enumerate(want["columns"]):
        assert t[col] == [row[j] for row in want["rows"]]


def test_tsv_ragged_row_is_refused_with_its_line(formats_dir):
    with pytest.raises(nn.FormatError, match="line"):
        nn.io.read_tsv(formats_dir / "data" / "tsv_ragged.tsv")


def test_fif_equals_the_mne_goldens(golden, f32, formats_dir):
    """Every real-valued FIF case of the goldens (MNE-Python): the header's channels and rate, the samples by block, the gaps of a skip, the parts of a split file."""
    seen = 0
    for name, case in golden("fif")["cases"].items():
        if not case["data"].endswith(".f32") or case.get("data_has_projector"):
            continue      # complex and float64 files are checked below; an active projector is the importer's to apply
        rec = nn.io.open_fif(formats_dir / "data" / case["file"])
        h = rec.header
        assert h["format"] == "fif" and h["precision"] == "float32"
        assert h["sampling_rate"] == pytest.approx(case["sfreq"]) and h["n_samples"] == case["n_times"]
        assert [c["name"] for c in h["channels"]] == case["info"]["ch_names"]
        assert sorted(h["bad_channels"]) == sorted(case["info"]["bads"])
        x = rec.read_all()
        np.testing.assert_allclose(x, f32(case["data"]).reshape(len(h["channels"]), case["n_times"]), rtol=2e-6, atol=1e-30, err_msg=name)
        a, b = case["n_times"] // 3, 2 * case["n_times"] // 3
        np.testing.assert_array_equal(rec.read(a, b), x[:, a:b])
        assert h["gaps"] == [(g["first"], g["count"]) for g in case["gaps"]]
        seen += 1
    assert seen >= 6


def test_fif_float64_and_refusals(formats_dir):
    d = formats_dir / "data"
    assert nn.io.open_fif(d / "fif_double_raw.fif").read_all().dtype == np.float64
    assert nn.io.open_fif(d / "fif_basic_raw.fif").header["info"]["first_samp"] == "250"
    with pytest.raises(nn.UnsupportedError):
        nn.io.open_fif(d / "fif_refuse_epo.fif")
    with pytest.raises(nn.UnsupportedError):
        nn.io.open_fif(d / "fif_complex_raw.fif").read(0, 10)      # complex buffers: refused by the real read
    with pytest.raises(nn.FormatError):
        nn.io.open_fif(d / "fif_refuse_notfif.fif")
    x, header = nn.io.read_fif(d / "fif_basic_raw.fif", start=10, stop=30)
    assert x.shape == (10, 20) and header["format"] == "fif"
