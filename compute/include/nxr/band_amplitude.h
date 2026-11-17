#pragma once
// nxr::band_amplitude (compute) — the amplitude of a spectrum in bands: the mean |X| over each band's bins, per row. Arrays in, arrays out.
#include <Eigen/Dense>
#include <cstdint>
#include <utility>
#include <vector>

namespace nxr {

/** the bins of one band: [first, last), counted from bin 0 of the spectrum's columns */
using BinRange = std::pair<std::int64_t, std::int64_t>;

/** spectrum: complex [rows x bins]; ranges: one [first, last) per band. Returns [rows x bands]: the mean of |X| over the band's bins.
 *  std::invalid_argument for a range that is empty or outside [0, bins]. */
Eigen::MatrixXd bandAmplitude(const Eigen::MatrixXcd& spectrum, const std::vector<BinRange>& ranges);

}  // namespace nxr
