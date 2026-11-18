// nxr::measurement (compute) — `estimate`: a statistic from a partition of the stream (spec docs/specs/2026-11-17-consolidation.md, section 2.5). Welch's PSD is the MEAN, over a
// family of segments (they may overlap), of a per-segment quantity, folded block by block; a distribution (the core threshold's histogram) is counts in fixed bins. The per-segment
// quantity is a pointwise transform (taper and FFT; X X^H; |X|^2; the mode weights), declared as nodes: `estimate` only means or counts. `welch::Welch`, `ComplexWelch`, `LinearWelch`
// and `CrossWelch` are compositions of the transform and `SegmentMean` (one accumulation).
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include "nxr/stats/histogram.h"

namespace nxr::measurement {

/** the mean over a family of segments of a per-segment quantity. A segment's value is added ONCE, complete (a segment carried across chunk edges is assembled by its owner and added when
 *  whole); a segment cut by an excluded span is `skip`ped and counted, never averaged. The sum is accumulated in the order segments are added, so the result is identical to the last bit
 *  however the record was cut into chunks. */
struct SegmentMean {
  std::int64_t used = 0, skipped = 0;
  /** add a complex quantity (X X^H: its real part is Re, its imaginary part Im); std::invalid_argument for a segment added twice or a shape that differs from the first */
  void add(std::int64_t segment, const Eigen::Ref<const Eigen::MatrixXcd>& value);
  /** add a real quantity (|X|^2): held in the real part of the sum */
  void add_real(std::int64_t segment, const Eigen::Ref<const Eigen::MatrixXd>& value);
  /** count a segment left out (it touched an excluded span) */
  void skip(std::int64_t segment);
  /** the sum of the segments added (what `mean` divides by `used`): the Welch classes scale it by scale / used as the reference does */
  const Eigen::MatrixXcd& sum() const { return sum_; }
  /** sum / used; std::invalid_argument when no segment was added */
  Eigen::MatrixXcd mean() const;
  /** real parts of mean() */
  Eigen::MatrixXd mean_real() const;

 private:
  Eigen::MatrixXcd sum_;
  std::set<std::int64_t> seen_;
  void note(std::int64_t segment);
};

/** counts in FIXED bins. `edges` are the bins' upper edges, ascending: bin k holds the values below edges[k] and at least edges[k - 1] (bin 0 everything below edges[0]); the last bin also
 *  holds what is above. `log_bins` makes the reference's log bins (nxr's `_core_threshold`: histogram::LogBins; the same edges as np.geomspace(lo, hi, bins), binned by the reference's
 *  own float32 formula so a value lands in the reference's bin; a negative or non-finite value is counted and left out, a non-negative series). */
struct Histogram {
  std::vector<double> edges;
  std::vector<std::int64_t> counts;
  std::int64_t negative = 0, nonfinite = 0;
  std::optional<stats::histogram::LogBins> log;
  static Histogram log_bins(const stats::histogram::LogBins& bins);
  /** counts the n values into the bins (additive: a distribution read block by block is the blocks' counts added) */
  void add(const float* v, std::int64_t n);
};

}  // namespace nxr::measurement
