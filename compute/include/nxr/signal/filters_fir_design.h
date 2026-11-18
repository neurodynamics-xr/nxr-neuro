#pragma once
// nxr::signal — FIR filter DESIGN by the window method: `fir1`, as MATLAB's fir1 and SciPy's scipy.signal.firwin.
//
// PURE MATHS: numbers in, the taps out. Conventions are MATLAB's: the ORDER n (n + 1 taps, a linear-phase filter, delay n / 2 samples), the edge(s) Wn in units of the
// NYQUIST frequency (0 < Wn < 1), the ideal response's impulse response multiplied by a window (default Hamming, symmetric, as fir1), and the taps scaled so the
// passband's reference has unit gain (DC for a low-pass and a band-stop, Nyquist for a high-pass, the centre of the first passband for a band-pass: fir1's default
// scaling, firwin's scale=True). A high-pass and a band-stop pass the Nyquist frequency, which an even number of taps (odd order) cannot: refused, as firwin does.
#include "nxr/signal/filters_iir.h"   // Band
#include <Eigen/Dense>
#include <vector>

namespace nxr::signal {

/** The taps (order + 1 of them) of the window-method filter. `Wn`: one edge (Lowpass, Highpass) or two ascending (Bandpass, Bandstop), in units of Nyquist.
 *  `window`: order + 1 samples (a symmetric window, e.g. windowOf(order + 1, Window::Hann, false)); empty: Hamming. `scale` false: the raw windowed ideal response (fir1 'noscale').
 *  Throws std::invalid_argument on a bad order, edge or window length. */
Eigen::VectorXd fir1(int order, const std::vector<double>& Wn, Band type, const Eigen::VectorXd& window = Eigen::VectorXd(), bool scale = true);

}  // namespace nxr::signal
