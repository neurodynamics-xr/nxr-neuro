"""The event, wavelet and filterbank functions: a call, its shapes, names and errors (the maths is the library's, tested there). Synthetic data only."""

import numpy as np
import pytest

import nxr_neuro as nn

sel = nn.compute.select
sg = nn.compute.signal


def planted_burst():
    t = np.arange(1000)
    x = 1 + 0.1 * np.sin(2 * np.pi * t / 37)
    x[400:460] = 3.0
    return x


def test_detect_bursts_finds_the_planted_burst():
    r = sel.detect_bursts(planted_burst(), 0.01)
    assert r.spans.start.tolist() == [400] and r.spans.stop.tolist() == [460]
    assert r.spans.peak.dtype == np.int64 and r.spans.value[0] == 3.0
    assert r.skipped == "" and r.good == 1000 and r.min_length == 40 and r.gap == 11
    assert r.median == pytest.approx(sel.median(planted_burst())) and r.sd == pytest.approx(sel.mad(planted_burst()))
    given = sel.detect_bursts(planted_burst(), 0.01, baseline=(r.median, r.sd))
    assert given.spans.start.tolist() == [400] and given.enter == pytest.approx(r.enter)


def test_crossings_merge_and_keep_compose_to_the_same_burst():
    x = planted_burst()
    r = sel.detect_bursts(x, 0.01)
    spans = sel.crossings(x, r.enter, r.exit)
    assert spans.start.tolist() == [400] and spans.stop.tolist() == [460]
    # a dip inside the burst splits it; merging closes it, and keeping drops the short pieces
    y = x.copy()
    y[430:435] = 1.0
    split = sel.crossings(y, r.enter, r.exit)
    assert len(split.start) == 2
    merged = sel.merge_spans(split, 11)
    assert merged.start.tolist() == [400] and merged.stop.tolist() == [460]
    assert len(sel.keep_spans(split, 40).start) == 0 and len(sel.keep_spans(merged, 40, 60).start) == 1


def test_risefall_and_stages():
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    a = 1 + 4 / ((1 + np.exp(-(t - 8) / 0.3)) * (1 + np.exp((t - 12) / 0.3)))
    f = sel.risefall(a, fs, 6.0, 14.0, min_r2=0.9)
    assert f.ok and f.kept is True and f.t_on == pytest.approx(8.0, abs=0.2) and f.t_off == pytest.approx(12.0, abs=0.2)
    assert sel.risefall(a, fs, 6.0, 14.0).kept is None
    s = sel.stages(f.t_on, f.t_off, f.tau_rise, f.tau_fall)
    assert s.valid and s.onset[0] < f.t_on < s.onset[1]
    with pytest.raises(ValueError, match="posterior"):
        sel.risefall(a, fs, 6.0, 14.0, model="posterior")


def test_band_envelope_of_a_sine_is_flat_inside():
    fs = 256.0
    t = np.arange(2048) / fs
    x = np.sin(2 * np.pi * 10 * t)[None, :]
    e = sg.band_envelope(x, fs, [(8, 12), (20, 30)])
    assert e.shape == (2, 1, 2048) and e.dtype == np.float64
    inner = e[0, 0, 256:-256]
    assert inner.std() / inner.mean() < 0.02 and inner.mean() == pytest.approx(1.0, rel=0.1)
    assert e[1, 0, 256:-256].max() < 0.05 * inner.mean()
    z = sg.band_analytic(x, fs, np.array([[8.0, 12.0]]), shape="meyer")
    assert z.shape == (1, 1, 2048) and np.iscomplexobj(z)
    assert sg.band_envelope(x.astype(np.float32), fs, [(8, 12)]).dtype == np.float32


def test_time_bank_describes_the_bank():
    b = sg.time_bank(256.0, 1024, "meyer")
    assert b.gains.shape == (len(b.centre_frequencies), len(b.frequencies)) and len(b.frequencies) == 513
    assert b.frame_bounds[0] == pytest.approx(b.frame_bounds[1])
    c = sg.time_bank(256.0, 1024, "morlet", sides="two_sided", centres=[6.0, 12.0], cycles=6, gain=2.0)
    assert c.gains.shape == (4, 1024) and c.frequencies.min() < 0   # a member at +f and one at -f per centre


def test_wt_iwt_dual_round_trip():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 1024))
    bank = {"family": "meyer", "sides": "analytic", "gain": 1.0}
    w = sg.wt(x, 256.0, bank)
    assert w.ndim == 3 and w.shape[1:] == (3, 1024) and np.iscomplexobj(w)
    y = sg.iwt(w, 256.0, 1024, bank)
    assert y.shape == x.shape and np.isrealobj(y)
    np.testing.assert_allclose(y, x, atol=1e-9)
    two = {"family": "meyer", "sides": "two_sided"}
    z = x + 1j * rng.standard_normal(x.shape)
    np.testing.assert_allclose(sg.iwt(sg.wt(z, 256.0, two), 256.0, 1024, two), z, atol=1e-9)


def test_graph_bank_weyl_has_levels_plus_one_members():
    b = sg.graph_bank("weyl", area=4 * np.pi, levels=4)
    assert len(b.roles) == 5 and b.roles[0] == "scaling" and b.roles[-1] == "top" and b.gains is None and np.isnan(b.frame_bounds[0])
    lam = np.linspace(0.0, 40.0, 200)
    g = sg.graph_bank("weyl", eigenvalues=lam, area=4 * np.pi, levels=4)
    assert g.gains.shape == (5, 200) and g.frame_bounds[0] == pytest.approx(1.0) and g.frame_bounds[1] == pytest.approx(1.0)
    h = sg.graph_bank("itersine", eigenvalues=lam, lmax=40.0, num_filters=4)
    assert h.gains.shape[1] == 200 and np.isfinite(h.centre_wavenumbers).all()


def path_laplacian(n):
    import scipy.sparse as sp
    d = np.ones(n - 1)
    L = sp.diags([-d, np.r_[1.0, 2 * np.ones(n - 2), 1.0], -d], [-1, 0, 1], format="csc")
    return L


def test_gwt_igwt_round_trip_on_a_path_graph():
    n = 40
    L = path_laplacian(n)
    lam, phi = np.linalg.eigh(L.toarray())
    F = np.random.default_rng(2).standard_normal((n, 5))
    bank = {"family": "weyl", "area": 2.0 * n, "levels": 3}
    w = sg.gwt(F, bank, eigenvalues=lam, eigenvectors=phi)
    assert w.shape == (4, n, 5)
    np.testing.assert_allclose(sg.igwt(w, bank, eigenvalues=lam, eigenvectors=phi), F, atol=1e-9)
    # the Chebyshev series stands for the exact transform
    ser = sg.gwt(F, {"family": "itersine", "lmax": 4.2, "num_filters": 4}, laplacian=L, lmax=4.2, order=60)
    ex = sg.gwt(F, {"family": "itersine", "lmax": 4.2, "num_filters": 4}, eigenvalues=lam, eigenvectors=phi)
    np.testing.assert_allclose(ser, ex, atol=1e-3)


def test_jwt_shape_and_modes_round_trip():
    n_t, fs = 512, 128.0
    lam = np.linspace(0.0, 4.0, 8)
    C = np.random.default_rng(3).standard_normal((8, n_t))
    g = {"family": "itersine", "lmax": 4.0, "num_filters": 3}
    t = {"family": "meyer", "sides": "analytic"}
    r = sg.jwt(C, fs, g, t, eigenvalues=lam)
    nb = len(sg.graph_bank("itersine", lmax=4.0, num_filters=3).roles) * len(sg.time_bank(fs, n_t, "meyer").centre_frequencies)
    assert r.coefficients.shape == (nb, 8, n_t) and r.index.shape == (nb, 3)
    assert r.index[0].tolist() == [0, 0, 0] and r.index[1, 0] == 1 and r.index[1, 1] == 0   # g varies fastest
    np.testing.assert_allclose(sg.ijwt(r.coefficients, fs, n_t, g, t, eigenvalues=lam), C, atol=1e-8)
    # Morlet x Meyer on a two-sided bank, with a speed kernel
    tm = {"family": "morlet", "sides": "two_sided", "centres": [4.0, 8.0], "cycles": 6.0, "gain": 2.0}
    z = C + 1j * C[::-1]
    k = sg.jwt(z, fs, g, tm, eigenvalues=lam, kernels=[{"type": "ridge", "v": 1.0, "width": 2.0, "sense": 1}])
    assert k.coefficients.shape[1:] == (8, n_t) and k.coefficients.shape[0] == k.index.shape[0] == 3 * 4 * 1


def test_names_the_library_refuses_are_value_errors():
    with pytest.raises(ValueError, match="nonsense"):
        sg.graph_bank("nonsense", lmax=1.0)
    with pytest.raises(ValueError, match="nonsense"):
        sg.time_bank(256.0, 256, "nonsense")
    with pytest.raises(ValueError, match="sideways"):
        sg.time_bank(256.0, 256, "meyer", sides="sideways")
    x = np.zeros((1, 256))
    with pytest.raises(ValueError, match="oblique"):
        sg.iwt(sg.wt(x, 256.0, {"family": "meyer"}), 256.0, 256, {"family": "meyer"}, synthesis="oblique")
    with pytest.raises(ValueError, match="shape"):
        sg.band_envelope(x, 256.0, [(8, 12)], shape="triangle")
    with pytest.raises(ValueError, match="no setting"):
        sg.wt(x, 256.0, {"family": "meyer", "colour": 3})
