import numpy as np
import pytest
import scipy.signal as sps

import nxr_neuro as nn

welch = nn.compute.welch


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
@pytest.mark.parametrize("detrend,sp_detrend", [("constant", "constant"), ("linear", "linear"), ("none", False)])
def test_psd_equals_scipy(signal, scaling, detrend, sp_detrend):
    x, fs = signal
    r = signal.psd(x, fs, segment_samples=256, overlap=0.5, detrend=detrend, scaling=scaling)
    f, p = sps.welch(x, fs, window="hann", nperseg=256, noverlap=128, detrend=sp_detrend, scaling=scaling, axis=-1)
    np.testing.assert_allclose(r.frequencies, f, rtol=1e-12)
    np.testing.assert_allclose(r.psd, p, rtol=1e-9, atol=1e-18)
    assert r.segments == (x.shape[1] - 256) // 128 + 1 and r.segments_skipped == 0


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "rectangular"])
def test_psd_windows(signal, window):
    x, fs = signal
    r = signal.psd(x, fs, segment_samples=512, overlap=0.25, window=window)
    name = {"rectangular": "boxcar"}.get(window, window)
    f, p = sps.welch(x, fs, window=name, nperseg=512, noverlap=128, detrend="constant", axis=-1)
    np.testing.assert_allclose(r.psd, p, rtol=1e-9, atol=1e-18)


def test_psd_precision_follows_the_input(signal):
    x, fs = signal
    r64 = signal.psd(x, fs)
    r32 = signal.psd(x.astype(np.float32), fs)
    assert r64.psd.dtype == np.float64 and r32.psd.dtype == np.float32
    np.testing.assert_allclose(r32.psd, r64.psd, rtol=2e-4)
    # an integer input is computed as float64
    xi = np.round(x * 100).astype(np.int64)
    assert signal.psd(xi, fs).psd.dtype == np.float64


def test_psd_bad_samples_leave_segments_out(signal):
    x, fs = signal
    good = np.ones(x.shape[1], dtype=bool)
    good[1000:1010] = False
    r = signal.psd(x, fs, segment_samples=256, overlap=0.5, good=good)
    full = signal.psd(x, fs, segment_samples=256, overlap=0.5)
    # segments start at multiples of 128; those touching samples 1000..1009 are 768 (768..1023), 896 (896..1151) and 1024? (no: 1024 > 1009)
    assert r.segments_skipped == 2 and r.segments == full.segments - 2


def test_psd_refuses_what_it_cannot_do(signal):
    x, fs = signal
    with pytest.raises(ValueError, match="window"):
        signal.psd(x, fs, window="triangle")
    with pytest.raises(ValueError):
        signal.psd(x, fs, good=np.ones(10, dtype=bool))
    with pytest.raises(TypeError):
        signal.psd(x[0], fs)  # one dimension: the signal is [channels, samples]


def _reference_cross(x, fs, n=256, overlap=0.5):
    hop = n - int(n * overlap)
    w = sps.get_window("hann", n, fftbins=True)
    segs = [x[:, s:s + n] for s in range(0, x.shape[1] - n + 1, hop)]
    acc = 0
    for s in segs:
        s = s - s.mean(axis=1, keepdims=True)
        X = np.fft.rfft(s * w, axis=1)
        acc = acc + X[:, None, :] * np.conj(X[None, :, :])      # [c, c, bins]: X_i conj(X_j)
    C = acc / len(segs) / (fs * np.sum(w ** 2))
    C[..., 1:-1] *= 2.0                                          # one-sided, n even
    return C


def test_cross_spectrum_equals_a_numpy_reference(signal):
    x, fs = signal
    x = x[:3]
    bins = [(0, 0), (12, 12), (30, 40)]
    r = welch.xpsd(x, fs, bins, segment_samples=256, quadrature=True)
    C = _reference_cross(x, fs)
    for g, (a, b) in enumerate(bins):
        want = C[..., a:b + 1].mean(axis=-1)
        np.testing.assert_allclose(r.real[g], want.real, rtol=1e-9, atol=1e-18)
        np.testing.assert_allclose(r.imag[g], want.imag, rtol=1e-9, atol=1e-18)
    assert r.real.shape == (3, 3, 3)
    np.testing.assert_allclose(r.imag, -np.swapaxes(r.imag, 1, 2), atol=1e-18)   # antisymmetric


def test_cross_spectrum_diagonal_is_the_psd(signal):
    x, fs = signal
    bins = [(k, k) for k in range(129)]
    r = welch.xpsd(x, fs, bins, segment_samples=256)
    p = signal.psd(x, fs, segment_samples=256).psd
    np.testing.assert_allclose(np.stack([np.diag(r.real[k]) for k in range(129)], axis=1), p, rtol=1e-10)
    assert r.imag is None


def test_cross_spectrum_sum_versus_mean(signal):
    x, fs = signal
    a = welch.xpsd(x, fs, [(10, 14)], mean=True).real
    b = welch.xpsd(x, fs, [(10, 14)], mean=False).real
    np.testing.assert_allclose(b, a * 5, rtol=1e-12)


def test_ops_is_a_deprecated_alias_of_compute():
    with pytest.warns(DeprecationWarning, match="nxr_neuro.compute"):
        ops = nn.ops
    assert ops.signal is nn.compute.signal
    with pytest.raises(AttributeError, match="renamed: use nxr_neuro.compute.signal"):
        nn.compute.welch
