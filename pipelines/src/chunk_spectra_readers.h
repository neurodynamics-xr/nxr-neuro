// nxr-pipelines (internal) — what the readers of the chunk spectra share (sensor_flow_steps.cpp defines them; the sensor-flow and the source-flow nodes read them):
// the chunk spectra [chunks x bins x coordinates] as signal.chunk_spectra stores them, a kernel set (the one reader) and its gains on a line, the frequency weighting's members,
// and one frequency member's weighted columns Y (the cross-spectrum it weights is Y Yᴴ).
#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "nxr/graph/node.h"
#include "nxr/selection/selection.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/kernel_gains.h"

namespace nxr::pipelines::graph::spectra_readers {

struct Spectra {
  std::int64_t chunks = 0, bins = 0, rows = 0, first = 0, transform = 0, overlap = 0;
  double sfreq = 0;
  std::vector<std::int64_t> modes, owns, firsts;   // per chunk: its own samples and its first sample on the recording's line
  bool unit_mass = true;
  const nxr::graph::Array* array = nullptr;
  std::complex<double> at(std::int64_t t, std::int64_t k, std::int64_t r) const {
    const std::size_t i = static_cast<std::size_t>(((t * bins + k) * rows + r) * 2);
    return std::complex<double>(array->at(i), array->at(i + 1));
  }
  double frequency(std::int64_t k) const { return static_cast<double>(first + k) * sfreq / static_cast<double>(transform); }
};

Spectra spectra_of(const nxr::graph::Array& spectra, const char* who);
/** the field coordinates' synthesis [S x K] (the constant at unit star0 norm; each mode at unit star0 norm under unit_mass) and their eigenvalues */
std::pair<Eigen::MatrixXd, Eigen::VectorXd> synthesis_of(const Spectra& s, const nxr::signal::Modes& m, const Eigen::VectorXd& mass, const char* who);
/** THE ONE READER of a stored kernel set (select.kernels) handed to a node: its members as a typed selection (spans on axis 0, each with its Profile). Refuses a
 *  set that is not a kernel set, and a member made before the proportional width was stored as fwhm_per_centre (re-make it with `select_kernels`). */
nxr::selection::Selection kernelSetOf(const nxr::graph::Array& stored, const char* who, const char* port);
/** its gains on points of its line's coordinate (Hz on a frequency line, mu on an eigenvalue line), by nxr::signal::kernelGains (a coordinate <= 0 has gain 0) */
nxr::signal::KernelGains kernel_gains_of(const nxr::graph::Array& stored, const Eigen::VectorXd& line, const char* who, const char* port);
/** the frequency weighting's members [members x bins] (parameters frequency_weighting, frequency_normalisation, support_floor; the port `frequency`) */
Eigen::MatrixXd frequency_gains(const Spectra& s, const nxr::graph::Arrays& c, const nxr::graph::Json& p, const char* who, std::vector<double>* standardDeviations = nullptr);
Eigen::MatrixXcd weighted_columns(const Spectra& s, const Eigen::RowVectorXd& gain, double floorFraction = 1e-12);

// ── THE TIME CELLS ON A STORED TIME GRID (spec docs/specs/2026-11-26-graph-manifold-and-time-cells.md, G7) ─────────────────────────────────────────
// A flow's time cells are the tiles of a STORED time grid (`select_time_grid --rule tiles`: one level k of the line's dyadic tiling, tiles of 2^k samples from the line's first
// sample), the same for every member. Each member's sums per tile are a MEASUREMENT over the product of the grid with the members (`select_product --time-grid --atoms
// [--scales]`: the grid's tile fastest), declared `sum`, so a coarser level is a roll-up at read. A windowed quantity (the sign-flip test's cells) uses each atom's OWN width:
// a box of `cell_width` time standard deviations snapped to the nearest dyadic level k_a >= k, read as the sums of 2^(k_a - k) consecutive stored tiles.
struct TimeGrid {
  std::int64_t width = 0, count = 0, size = 0;   // the tile (2^level samples), the tiles, the line's samples
  int level = 0;
  double rate = 0;
  std::string name, id;                          // id: the stored grid's row (empty when made in the run)
  nxr::graph::Json record() const;
};
/** the stored grid handed to a node (its params.grid), checked against the spectra's line (rate, samples); refused for a grid that is not one dyadic level from the line's first sample */
TimeGrid time_grid_of(const nxr::graph::Array& grid, const Spectra& s, const char* who);
/** an atom's own window level: `cellWidth` time standard deviations of an atom whose frequency standard deviation is `frequencySd` (Hz), the nearest power of two of samples;
 *  refused when it is finer than the grid's level (name the grid that is needed) */
int own_level(double cellWidth, double frequencySd, const TimeGrid& g, const char* who);
/** the product the cells are measured over (`select_product`), checked: its factors' radices are the grid's tiles and then `members` */
void check_cells_product(const nxr::graph::Array& product, const TimeGrid& g, const std::vector<std::int64_t>& members, const char* who);
/** one member's sums per grid tile over every chunk of the spectra: the member's gains on the bins, back in time through `readout` (timeCellPairFlow); a tile cut by a chunk's
 *  edge completed from the next chunk; the line's last tile as long as it is. [rows x tiles] each; vertexCurrent empty without `sharp` */
struct TileSums { Eigen::MatrixXd power, pairReal, pairImaginary, vertexCurrent; std::vector<std::int64_t> samples; };
TileSums tile_sums(const Spectra& s, const Eigen::RowVectorXd& gains, const TimeGrid& g, const Eigen::MatrixXd& readout, int components, const std::vector<std::int32_t>& pairs,
                   const Eigen::SparseMatrix<double>* sharp);
/** a cells MEASUREMENT over the product (rows x members x measures, members = tile + tiles x group): `values[g][k]` the group's [rows x tiles] of measure k (stored float32);
 *  `groups` [{atom, band, own_level, ...}] in the product's member order; declared by flow_measures.h */
nxr::graph::Array cells_measurement(const std::vector<std::vector<Eigen::MatrixXd>>& values, const std::vector<std::string>& measures, const TimeGrid& g, const nxr::graph::Json& groups,
                                    const char* rowsAre, const nxr::graph::Json& producer);
/** a cells measurement read back (made in the run or as stored): the grid, its groups and their measures; `rolled` one group's measure summed over each whole cell of the
 *  group's own level ([cells x rows]: cell c the tiles c 2^(k_a - k) .. (c + 1) 2^(k_a - k) - 1, by nxr::measurement::coarser_groups; a cell past the line's end left out) */
struct StoredCells {
  const nxr::graph::Array* array = nullptr;
  std::int64_t rows = 0, members = 0, K = 0, tiles = 0, width = 0, size = 0;
  int level = 0;
  nxr::graph::Json groups, measures;
  Eigen::MatrixXd rolled(std::int64_t group, std::int64_t measure) const;
  std::int64_t measure_index(const std::string& name, const char* who) const;
};
StoredCells stored_cells(const nxr::graph::Array& cells, const char* who);

}  // namespace nxr::pipelines::graph::spectra_readers
