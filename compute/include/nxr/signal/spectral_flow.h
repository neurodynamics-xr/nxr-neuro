#pragma once
// nxr::signal — THE FLOW OF A WEIGHTED CROSS-SPECTRUM in a surface's field coordinates (spec docs/specs/2026-11-24-sensor-flow.md). PURE MATHS: arrays in, arrays out;
// nothing here knows a file, a chunk or a subject.
//
//   overlappedSpectrum      one chunk with its overlap (samples taken in from the neighbouring chunks) → the serving operator's rows, then ONE Fourier transform of
//                           `transformSamples` samples (no window, no taper, no zero padding), the bins [firstBin, firstBin + binCount) kept. A side the samples do not
//                           reach (a recording's end) is the samples reflected about their outermost sample (numpy `reflect`: the edge sample not repeated)
//   gaussianAtomGains       Gaussian gains on a frequency line: gain · exp(−(f − centre)² / (2 sd²)) for f > 0, 0 elsewhere (one-sided: the inverse transform is the analytic signal)
//   normaliseBySumOfSquaredGains   gains [members x points] divided at each point by the square root of the set's sum of squared gains there, so the squared gains sum to one
//                           wherever that sum is at least `floor` x its largest value (0 elsewhere: outside the set's support)
//   cosineSquaredPartition  an AMPLITUDE partition on a line given in a log2 chart: member b rises as sin² from centre b−1 to centre b and falls as cos² from centre b to b+1;
//                           the first member is 1 below its centre, the last 1 above its centre (the complement), so the gains sum to one at every point. A point at
//                           −infinity (a constant, mu 0) belongs to the first member
//   mexicanHatPartition     an AMPLITUDE partition on an eigenvalue line from the graph bank's Mexican hats in lambda = mu², each divided by the members' sum at the point
//   bandPairFlow            per pair (p, q) of eigenvalue bands (gains h [bands x K]) the second-order readings of a cross-spectrum C [K x K] in field coordinates:
//                           power per vertex Re(Psi_v h_p conj(C) h_q Psi_vᵀ), phase current per edge Im(Psi_tail h_p conj(C) h_q Psi_tipᵀ) (= Σ Im(conj(z_p,tail) z_q,tip)
//                           over the draws of C, `phaseCurrent`'s definition) and phase current per vertex Im(Psi_v h_p conj(C) h_q G_vᵀ) (3 components: the
//                           power-weighted phase gradient's numerator). The pairs of an amplitude partition sum to the whole field's values.
//   timeCellFlow            one chunk's weighted spectrum back in time per band, summed over cells of a level of the time line: per cell the edge phase current
//                           Σ Im(conj(z_tail) z_tip), the vertex phase current Σ Im(conj(z_v) grad z_v) and the power Σ |z_v|² (a measurement over the cells, the
//                           per-sample products never held beyond one chunk)
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::signal {

/** `servingOperator` [K x channels], `samples` [channels x (left + own + right)] (`left`, `right` the samples present beyond the chunk's `own`), `overlap` the samples the transform takes
 *  on the left, `transformSamples` its length (the right side gets transformSamples − overlap − own). Throws std::invalid_argument when a side needs more reflection than the samples hold,
 *  or the bins fall outside the transform's. Returns [K x binCount] complex, unscaled (numpy rfft norm backward). */
Eigen::MatrixXcd overlappedSpectrum(const Eigen::MatrixXd& servingOperator, const Eigen::MatrixXd& samples, std::int64_t left, std::int64_t right, std::int64_t overlap,
                                    std::int64_t transformSamples, std::int64_t firstBin, std::int64_t binCount);

/** [centres x frequencies]. Throws std::invalid_argument on centres and deviations of different counts or a deviation <= 0. */
Eigen::MatrixXd gaussianAtomGains(const Eigen::VectorXd& frequencies, const std::vector<double>& centres, const std::vector<double>& standardDeviations, double gain);
Eigen::MatrixXd normaliseBySumOfSquaredGains(const Eigen::MatrixXd& gains, double floor);
/** `log2Coordinate` [points] (−infinity allowed), `log2Centres` ascending [members] (at least 2). [members x points] */
Eigen::MatrixXd cosineSquaredPartition(const Eigen::VectorXd& log2Coordinate, const std::vector<double>& log2Centres);
/** The MEXICAN HAT amplitude partition on an eigenvalue line: member b's hat in the eigenvalue lambda = mu², g_b = (lambda / c_b) e^(1 − lambda / c_b) with c_b = (its centre
 *  wavenumber)² (the graph bank's Mexican hat t lambda e^(−t lambda), t = 1 / c_b, at unit peak: graphfilterbank::mexicanHat), DIVIDED at each point by the members' sum, so the
 *  gains sum to one at every mu > 0. Computed in the log domain (no underflow far above the last centre: there the last member tends to one). A point at mu = 0 (the constant,
 *  where every hat is 0) belongs to the first member. `wavenumbers` mu [points] (>= 0), `centreWavenumbers` ascending [members] (at least 2). [members x points] */
Eigen::MatrixXd mexicanHatPartition(const Eigen::VectorXd& wavenumbers, const std::vector<double>& centreWavenumbers);

struct BandPairFlow {
  Eigen::MatrixXd power;          // [S x pairs] pair p * bands + q
  Eigen::MatrixXd edgeCurrent;    // [E x pairs]
  Eigen::MatrixXd vertexCurrent;  // [3S x pairs] rows 3v + component
};
/** `crossSpectrum` C [K x K] Hermitian (a SUM of x xᴴ), `synthesis` Psi [S x K], `gradients` G [3S x K] (rows 3v + component: each coordinate's tangent gradient), `edges` [E x 2]
 *  (tail, tip), `bandGains` h [bands x K]. Cheapest order: per band q one product conj(C) diag(h_q) [Psiᵀ | Gᵀ], then per pair only the entries read. Throws on sizes that disagree. */
BandPairFlow bandPairFlow(const Eigen::MatrixXcd& crossSpectrum, const Eigen::MatrixXd& synthesis, const Eigen::MatrixXd& gradients, const std::vector<std::int32_t>& edges,
                          const Eigen::MatrixXd& bandGains);

struct CellFlow {
  Eigen::MatrixXd edgeCurrent;    // [E x cells]
  Eigen::MatrixXd vertexCurrent;  // [3S x cells]
  Eigen::MatrixXd power;          // [S x cells]
};
/** `weightedSpectrum` [K x binCount] (an atom's gains already applied) at bins [firstBin, firstBin + binCount) of a transform of `transformSamples`; the band signal
 *  z = ifft(Psi diag(bandGain) X) (the one-sided spectrum inverted as a complex signal: the analytic signal when the gains carry the factor 2), the samples
 *  [ownFirst, ownFirst + ownSamples) kept, cut into cells of `cellSamples` from ownFirst (a trailing part cell left out). Throws on sizes that disagree. */
CellFlow timeCellFlow(const Eigen::MatrixXcd& weightedSpectrum, std::int64_t firstBin, std::int64_t transformSamples, std::int64_t ownFirst, std::int64_t ownSamples,
                      std::int64_t cellSamples, const Eigen::MatrixXd& synthesis, const Eigen::MatrixXd& gradients, const std::vector<std::int32_t>& edges,
                      const Eigen::VectorXd& bandGain);

}  // namespace nxr::signal
