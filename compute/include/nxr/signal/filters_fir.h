#pragma once
// nxr::taps — ZERO-PHASE TAPS AND THE ROW CONVOLUTION: the maths of a pipeline's `band_filter` step and of its analytic signal. A line port of nxr's
// `nxr/filters/taps.py` (`gain_taps`, `convolve_rows`), which is the Studio's `gainTaps` and `convolveRows` (studio/src/studio/engine/gpu/deviceChain.ts),
// the reference: a real gain G(f) taken to the time domain ONCE (the inverse FFT of G sampled on a fine grid; the even taps, truncated where the tail holds
// under kTailEnergy of the response's energy, at most kMaxTaps a side; the DC restored so that Σ h = G(0)), then every row convolved with them, an index
// past either end of the block REFLECTED evenly about it.
//
// THE ANALYTIC SIGNAL (the Studio's device chain, "THE ANALYTIC SIGNAL"): the same gain's ODD taps are the taps of its Hilbert transform (−i·sign(f)·G;
// g[0] = 0), so the analytic signal of a band is two convolutions of the same raw rows: re = Σ h[k](x[u−k] + x[u+k]), im = Σ g[k](x[u−k] − x[u+k]). A pure
// cosine at a frequency the band passes comes out as e^{iωt}: a constant amplitude and a phase advancing at ω.
//
// THE HALO: the taps are finite (K a side), so a block read with K samples of context on each side gives, on its own columns, exactly what the whole
// recording gives there; at a recording's end the block's own edge is reflected, as the Studio's convolveRows does. What makes a filter a per-chunk step.
//
// PURE MATHS: arrays in, arrays out (the FFT is nxr::fft, numpy's pocketfft). Rows are row-major [C x Wr], float32 as the Studio's and nxr's.
#include <cstddef>
#include <functional>
#include <vector>

namespace nxr::signal {

constexpr std::size_t kMaxTaps = 1024;       // the most taps on each side of the centre (deviceChain.ts MAX_TAPS)
constexpr double kTailEnergy = 1e-6;         // the share of the response's energy the truncation may drop (TAIL_ENERGY)
constexpr std::size_t kGrid = 1u << 14;      // the grid the gain is sampled on (GRID)

/** the taps of a real gain G(f) (`gain` takes a frequency in Hz) at the rate `fd` (Hz): even h[0..K] (zero phase), or `odd` the taps of its Hilbert
 *  transform (g[0] = 0). nxr's `gain_taps`, the Studio's `gainTaps`. */
std::vector<float> gainTaps(const std::function<double(double)>& gain, double fd, bool odd = false);

/** rows x [C x Wr] (row-major) -> y [C x width], y[c, t] = Σ_k h[k] (x[c, u − k] ± x[c, u + k]), u = left + t (the block's own columns start at `left`);
 *  an index past either end of the block reflected evenly about it. `odd`: the Hilbert's antisymmetric sum. nxr's `convolve_rows`, the Studio's
 *  `convolveRows`. Throws std::invalid_argument when [left, left + width) is not inside the block. */
std::vector<float> convolveRows(const float* x, std::size_t C, std::size_t Wr, const std::vector<float>& taps, std::size_t left, std::size_t width, bool odd = false);

}  // namespace nxr::signal
