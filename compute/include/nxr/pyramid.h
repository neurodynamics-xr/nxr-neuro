#pragma once
// nxr::pyramid — the maths of the MEASUREMENT PYRAMID's band power. PURE MATHS: arrays in, arrays out — no file, no chunk, no recording, no dataset.
//
//   octaveBands       the default bank: octaves [1·2^b, 2·2^b) Hz, b = 0 … 5 (nxr DEFAULT_BANDS)
//   bandGains         [bands x frequencies] amplitude gains of a bank, two SHAPES:
//                       PowerPartition  nxr's `band_gains`: squared gains sum to 1 inside [first low, last high] (cos/sin crossover half an octave wide in log2 f), so band powers add to broadband (Parseval)
//                       Morlet          the project's fixed-band rule: a Gaussian, unit peak, centred on (lo + hi)/2, FWHM = hi − lo; neighbouring bands overlap
//   spectrumOf        a block's spectrum [samples x channels] (time down the rows); analyticBand — one band's analytic signal from it, zero-phase: positive frequencies × 2 g(|f|), the rest 0
//                     (nxr `analytic_bands`; DC and, for an even length, the Nyquist bin are dropped as there). The caller passes the block it wants filtered WITH its context (halo): the
//                     filter is circular over the block, so the edges of a block are not exact and the caller keeps the middle (the chunk service's halo). Filtering the whole record in one block IS nxr's.
//   frameOfSamples    the frame of each sample of a span: the member of a FRAME GRID resolved on the time line (selection::Intervals, one interval a frame: `resolve_intervals` of the
//                     tower's tiling, nxr/frame_grid.h), minus the frame the caller's accumulator starts at (spec docs/specs/2026-11-17-selection-module.md, section 8: the tile per sample)
//   addSampleCounts   the good samples of each frame (the pyramid's per-frame `samples`: a `count` on the frame grid), added into the caller's counts: additive over blocks
//   bandSums          J = K A of a band's analytic block (the inverse is linear, so the filter comes first, at the sensors), then per (leaf, frame) Σ a|J|², Σ a|J|, max|J| (partition::analyticSums):
//                     a measurement over the cells of the joint grid (leaves × frames); the leaves are a `MemberOfElement` (the member number of each row), the frames the member of each sample;
//                     bad samples carry J = 0 (nxr). Blocked over samples, so the memory held is  rows x block x 4 doubles, never rows x samples.
// A reduction across time or space is never a node's own: bandSums calls the partition measurement; what it accumulates is mergeable (sums add, maxima merge), so a result built block by block does
// not depend on the block size.
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "nxr/partition.h"
#include "nxr/selection/resolved.h"
#include "nxr/tower.h"

namespace nxr::pyramid {

using Bands = std::vector<std::pair<double, double>>;
enum class Shape { Morlet, PowerPartition };

/** Octaves [f_lo·2^b, f_lo·2^(b+1)), b = 0 … n − 1 (nxr `octave_bands`: 1–2 … 32–64 Hz by default). */
Bands octaveBands(double fLo = 1.0, int n = 6);
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

/** The frame (member number) of each sample [first, first + n) of the line, minus `frame0` (the frame the accumulator's first tile is). `frames` is a frame grid resolved on the line (one interval
 *  a member, sorted, disjoint). Throws std::invalid_argument for a sample in no frame or before `frame0`. */
std::vector<std::int32_t> frameOfSamples(const selection::Intervals& frames, std::int64_t first, std::int64_t n, std::int64_t frame0 = 0);
/** Adds the good samples [first, first + n) (`good` null: all good; else n entries, 0 = bad) to `counts` (one per frame of the grid, minus `frame0`). */
void addSampleCounts(const selection::Intervals& frames, std::int64_t first, std::int64_t n, const std::uint8_t* good, std::int64_t frame0, std::vector<std::int64_t>& counts);

/** Adds a band's cortical-current sums to `out`: A [n x channels] is the band's analytic signal at the sensors (the kernel's channels, in the kernel's order), K [rows x channels] the kernel
 *  (constrained orientation: one row a vertex); `good` (n entries, null: all good) zeroes bad samples before the projection; `leaves` the leaf (member number) of each row (-1: none),
 *  `weights` [rows] its area; `frame_of_sample` [n] the frame of each sample (`frameOfSamples`; < 0: none). `block` samples are projected at a time.
 *  `threads` (nxr::parallel::resolveThreads): the blocks' kernel products run on that many workers and the sums are folded in block order on the calling thread, so the result is BIT-IDENTICAL to threads = 1;
 *  memory: (threads + 1) blocks of products held at once (rows x block x 2 doubles each). */
void bandSums(partition::AnalyticSums& out, const Eigen::MatrixXd& K, const Eigen::MatrixXcd& A, const std::uint8_t* good, const selection::MemberOfElement& leaves, const double* weights,
              const std::vector<std::int32_t>& frame_of_sample, std::int64_t block = 512, int threads = 1);

// ── THE FREE-ORIENTATION TENSOR (nxr `measure.vectors.band_tensor`) ──
/** A free-orientation kernel [3V x M] (rows 3v + (x, y, z): Brainstorm's order) projected on the canonical frames: K1 = e1·K, K2 = e2·K, Kn = n·K, each [V x M]. `e1`, `e2`, `n` [V x 3] (NaN at a vertex with no
 *  frame), `singular` [V] marks those: as nxr, a singular vertex takes an ARBITRARY tangent basis (tangent_basis of its normal, (0, 0, 1) where that is NaN) and a zero normal row, so it is out of the nine tensor
 *  sums (its weight is 0) yet its tangential power is in `total`. Throws std::invalid_argument on a shape mismatch. */
struct FrameKernel { Eigen::MatrixXd K1, K2, Kn; };
FrameKernel frameKernel(const Eigen::MatrixXd& K3, const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2, const Eigen::MatrixXd& n, const std::vector<char>& singular, int threads = 1);
/** Adds a band's orientation tensor to `out`: A [n x channels] the band's analytic signal at the sensors; the six real products K1 A, K2 A, Kn A (K real) are blocked over `block` samples (memory: rows x block x 6 doubles per held block, (threads + 1) blocks with `threads` > 1; the result is BIT-IDENTICAL for every `threads`, the blocks' sums being folded in block order).
 *  `weights` [rows]: the area of a vertex with a frame, 0 of a singular one; `totalWeights` [rows]: the area of every vertex; `leaves`, `frame_of_sample`, `good` as bandSums. */
void tensorBandSums(partition::TensorSums& out, const FrameKernel& K, const Eigen::MatrixXcd& A, const std::uint8_t* good, const selection::MemberOfElement& leaves, const double* weights, const double* totalWeights,
                    const std::vector<std::int32_t>& frame_of_sample, std::int64_t block = 128, int threads = 1);

}  // namespace nxr::pyramid
