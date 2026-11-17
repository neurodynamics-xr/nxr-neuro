import numpy as np
import pytest

import nxr_neuro as nn

filt = nn.compute.filter
FS = 1000.0


def tones(freqs, n=8000):
    t = np.arange(n) / FS
    return np.stack([np.sin(2 * np.pi * f * t) for f in freqs])


def gain(y, x, lo=1500, hi=-1500):
    """Amplitude ratio of output to input away from the ends."""
    return np.sqrt(np.mean(y[:, lo:hi] ** 2, axis=1) / np.mean(x[:, lo:hi] ** 2, axis=1))


def test_bandpass_passes_the_band_and_stops_the_rest():
    x = tones([3, 15, 80])
    g = gain(filt.bandpass(x, 10, 20, FS), x)
    assert g[1] == pytest.approx(1.0, abs=0.02) and g[0] < 1e-2 and g[2] < 1e-2


def test_lowpass_and_highpass_split_the_spectrum():
    x = tones([5, 100])
    lo = gain(filt.lowpass(x, 30, FS), x)
    hi = gain(filt.highpass(x, 30, FS), x)
    assert lo[0] == pytest.approx(1.0, abs=0.02) and lo[1] < 1e-2
    assert hi[1] == pytest.approx(1.0, abs=0.02) and hi[0] < 1e-2


def test_notch_removes_the_line_and_its_harmonic():
    x = tones([10, 50, 100])
    g = gain(filt.notch(x, [50, 100], FS), x)
    assert g[0] > 0.98 and g[1] < 0.05 and g[2] < 0.05


def test_bandstop_removes_the_band():
    x = tones([5, 45, 120])
    g = gain(filt.bandstop(x, 40, 50, FS), x)
    assert g[0] > 0.98 and g[1] < 0.05 and g[2] > 0.98


def test_filters_are_zero_phase_and_keep_shape_and_precision(signal):
    x, fs = signal
    y = filt.bandpass(x, 8, 16, fs)
    assert y.shape == x.shape and y.dtype == np.float64
    y32 = filt.bandpass(x.astype(np.float32), 8, 16, fs)
    assert y32.dtype == np.float32
    np.testing.assert_allclose(y32, y, atol=2e-4 * np.abs(y).max())
    # zero phase: a symmetric pulse stays symmetric about its centre
    pulse = np.zeros((1, 4001))
    pulse[0, 2000] = 1.0
    out = filt.lowpass(pulse, 40, FS, remove_mean=False)[0]
    np.testing.assert_allclose(out[2000 - 300:2000], out[2000 + 300:2000:-1], atol=1e-12)


def test_options_are_named_and_checked():
    x = tones([5, 50])
    assert filt.bandstop(x, 40, 60, FS, method="iir").shape == x.shape
    assert filt.lowpass(x, 30, FS, edge="zeros").shape == x.shape
    with pytest.raises(ValueError, match="method"):
        filt.lowpass(x, 30, FS, method="elliptic")
    with pytest.raises(ValueError, match="IIR"):
        filt.bandpass(x, 10, 20, FS, method="iir")   # the library's IIR designs are the notch and the band-stop
    with pytest.raises(ValueError):
        filt.bandpass(x, 20, 10, FS)        # lower edge above the upper
    with pytest.raises(ValueError):
        filt.lowpass(x, 600, FS)            # above Nyquist
