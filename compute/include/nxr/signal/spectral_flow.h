#pragma once
// nxr::signal — THE FLOW OF A WEIGHTED CROSS-SPECTRUM in a surface's field coordinates (spec docs/specs/2026-11-24-sensor-flow.md). PURE MATHS: arrays in, arrays out;
// nothing here knows a file, a chunk or a subject.
//
//   overlappedSpectrum      one chunk with its overlap (samples taken in from the neighbouring chunks) → the serving operator's rows, then ONE Fourier transform of
//                           `transformSamples` samples (no window, no taper, no zero padding), the bins [firstBin, firstBin + binCount) kept. A side the samples do not
//                           reach (a recording's end) is the samples reflected about their outermost sample (numpy `reflect`: the edge sample not repeated)
//   normaliseBySumOfSquaredGains   gains [members x points] divided at each point by the square root of the set's sum of squared gains there, so the squared gains sum to one
//                           wherever that sum is at least `floor` x its largest value (0 elsewhere: outside the set's support)
//   cosineSquaredPartition  an AMPLITUDE partition on a line given in a log2 chart: member b rises as sin² from centre b−1 to centre b and falls as cos² from centre b to b+1;
//                           the first member is 1 below its centre, the last 1 above its centre (the complement), so the gains sum to one at every point. A point at
//                           −infinity (a constant, mu 0) belongs to the first member
//   (the flow of a weighted cross-spectrum through these coordinates and its time cells: flow_readout.h, pairFlow and timeCellPairFlow)
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::signal {

/** `servingOperator` [K x channels], `samples` [channels x (left + own + right)] (`left`, `right` the samples present beyond the chunk's `own`), `overlap` the samples the transform takes
 *  on the left, `transformSamples` its length (the right side gets transformSamples − overlap − own). Throws std::invalid_argument when a side needs more reflection than the samples hold,
 *  or the bins fall outside the transform's. Returns [K x binCount] complex, unscaled (numpy rfft norm backward). */
Eigen::MatrixXcd overlappedSpectrum(const Eigen::MatrixXd& servingOperator, const Eigen::MatrixXd& samples, std::int64_t left, std::int64_t right, std::int64_t overlap,
                                    std::int64_t transformSamples, std::int64_t firstBin, std::int64_t binCount);

Eigen::MatrixXd normaliseBySumOfSquaredGains(const Eigen::MatrixXd& gains, double floor);
/** `log2Coordinate` [points] (−infinity allowed), `log2Centres` ascending [members] (at least 2). [members x points] */
Eigen::MatrixXd cosineSquaredPartition(const Eigen::VectorXd& log2Coordinate, const std::vector<double>& log2Centres);

}  // namespace nxr::signal
