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

/** the bins of the band [lo_hz, hi_hz) of a spectrum of `tile_samples` samples at `sfreq` Hz (spacing df = sfreq / tile_samples): [ceil(lo / df), ceil(hi / df)), the top held
 *  inside the spectrum's tile_samples / 2 + 1 bins. std::invalid_argument for a band that holds no bin. */
BinRange bandBins(double lo_hz, double hi_hz, std::int64_t tile_samples, double sfreq);

/** the amplitude of every row of a block in ONE band, per time tile: `x` [rows x samples] is cut into tiles of `tile_samples` from its first sample (a trailing part shorter than a
 *  tile is left out), each tile's one-sided FFT is taken as it is (no window, no taper), and the amplitude is the mean |X| over the band's bins (`bandBins`). Returns [rows x tiles]:
 *  rows are the block's rows, columns the grid's members in time order. std::invalid_argument for a tile longer than the block or a band that holds no bin. */
Eigen::MatrixXd bandAmplitudeOverTiles(const Eigen::MatrixXd& x, std::int64_t tile_samples, double sfreq, double lo_hz, double hi_hz);

}  // namespace nxr
