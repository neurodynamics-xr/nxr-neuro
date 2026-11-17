#pragma once
// nxr::pyramid — the TRANSFORM maths of a band's analytic signal (the chain's pointwise stages). PURE MATHS: arrays in, arrays out — no file, no chunk, no recording, no dataset.
//
//   bandGains         [bands x frequencies] amplitude gains of a bank, two SHAPES:
//                       PowerPartition  nxr's `band_gains`: squared gains sum to 1 inside [first low, last high] (cos/sin crossover half an octave wide in log2 f), so band powers add to broadband (Parseval)
//                       Morlet          the project's fixed-band rule: a Gaussian, unit peak, centred on (lo + hi)/2, FWHM = hi − lo; neighbouring bands overlap
//   spectrumOf        a block's spectrum [samples x channels] (time down the rows); analyticBand — one band's analytic signal from it, zero-phase: positive frequencies × 2 g(|f|), the rest 0
//                     (nxr `analytic_bands`; DC and, for an even length, the Nyquist bin are dropped as there). The caller passes the block it wants filtered WITH its context (halo): the
//                     filter is circular over the block, so the edges of a block are not exact and the caller keeps the middle (the chunk service's halo). Filtering the whole record in one block IS nxr's.
//   frameKernel       a free-orientation kernel projected on the canonical frames (K1, K2, Kn), the linear map a cortical chain folds into its operator
// The reductions this file used to carry (band sums, tensor sums, the frame of a sample, the good samples of a frame) are `nxr::measurement::reduce` over a product selection
// (spec docs/specs/2026-11-17-consolidation.md): a reduction across time or space is never a function here. The octave bank is `selection::octave_bands` (selection/bands.h).
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nxr::pyramid {

using Bands = std::vector<std::pair<double, double>>;
enum class Shape { Morlet, PowerPartition };

Shape shapeOf(const std::string& name);              // "morlet" | "power_partition"; throws std::invalid_argument naming both
const char* shapeName(Shape s);

/** [bands x freqs] amplitude gains at |f| (freqs in Hz, any sign). PowerPartition: nxr `band_gains` (needs adjacent, ascending bands); Morlet: filterbank::morletBand of each band. */
Eigen::MatrixXd bandGains(const Eigen::VectorXd& freqs, const Bands& bands, Shape shape);

struct Spectrum {
  Eigen::MatrixXcd X;          // [n x channels], numpy's fft down the columns
  Eigen::VectorXd freq;        // [n] numpy fftfreq(n, 1/sfreq), Hz
  double sfreq = 0;
};
/** The spectrum of a real block [samples x channels]. */
Spectrum spectrumOf(const Eigen::MatrixXd& block, double sfreq);
/** The analytic signal [samples x channels] of the band with amplitude gain `gain` (one entry per bin of `spec.freq`, a function of |f|): ifft(X · 2 · gain · [f > 0]). */
Eigen::MatrixXcd analyticBand(const Spectrum& spec, const Eigen::VectorXd& gain);

// ── THE FREE-ORIENTATION KERNEL (nxr `measure.vectors.band_tensor`) ──
/** A free-orientation kernel [3V x M] (rows 3v + (x, y, z): Brainstorm's order) projected on the canonical frames: K1 = e1·K, K2 = e2·K, Kn = n·K, each [V x M]. `e1`, `e2`, `n` [V x 3] (NaN at a vertex with no
 *  frame), `singular` [V] marks those: as nxr, a singular vertex takes an ARBITRARY tangent basis (tangent_basis of its normal, (0, 0, 1) where that is NaN) and a zero normal row, so it is out of the nine tensor
 *  sums (its weight is 0) yet its tangential power is in `total`. Throws std::invalid_argument on a shape mismatch. */
struct FrameKernel { Eigen::MatrixXd K1, K2, Kn; };
FrameKernel frameKernel(const Eigen::MatrixXd& K3, const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2, const Eigen::MatrixXd& n, const std::vector<char>& singular, int threads = 1);

}  // namespace nxr::pyramid
