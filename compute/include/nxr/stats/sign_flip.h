#pragma once
// nxr::stats — THE SIGN-FLIP (TIME-REVERSAL) TEST OF A DIRECTION'S CONSISTENCY over the cells of a time level, and the random draws it uses. PURE MATHS on arrays.
// Reference: nxr's flow maps (`nxr.flow.flowmap.flip_test`), the reference flow analysis's sign-flip test: time reversal conjugates a cross-spectrum, so each cell's phase
// current changes sign; the cells' unit vectors are summed per element and their resultant length R1 compared with random sign patterns over the cells.
//
//   NumpyGenerator          numpy's `default_rng(seed)`: the SeedSequence of the seed, PCG64 (XSL-RR, 128-bit state), and `random()` = (next 64 bits >> 11) / 2^53.
//                           The same draws as numpy, bit for bit, so a statistic's p values reproduce a reference that drew with numpy (nxr's flow analysis's default_rng(0))
//   signFlipConsistency     values [cells x elements·components] (an element's components consecutive): U = each cell's vector cut to unit length per element (0 stays 0);
//                           R1_e = |Σ_cells U| / cells; `patterns` sign vectors over groups of `cellsPerFlip` consecutive cells, drawn as numpy's
//                           random((patterns, groups)) < 0.5 → −1 (else +1), in that order; per element p = (#(null R1 ≥ R1) + 1)/(patterns + 1); the family-wise
//                           p = (#(null max over elements ≥ R1) + 1)/(patterns + 1); the threshold = the null maximum's 95th percentile; the median R1 over elements,
//                           its null's 95th percentile and p (numpy's median and linear percentile). Also the mean correlation of neighbouring cells' unit fields
//                           (Σ_e U_s,e · U_s+1,e / elements, the mean over s): the check of whether single cells are exchangeable
//   groupNeighbourCorrelation   the unit fields summed over groups of `cellsPerGroup` consecutive cells (whole groups only), then the mean over neighbouring groups of the
//                           cosine of their sums (Σ G_g · G_g+1 / (|G_g| |G_g+1|)); one cell a group: the neighbouring cells' correlation above
//   fewestDecorrelatedCells THE GROUPING RULE: the fewest consecutive cells (1, 2, 3, ...) whose group sums' neighbour correlation is below `threshold`, searched while at least
//                           `minimumGroups` whole groups remain; the group a sign flips together, so the flipped units are practically uncorrelated (exchangeable)
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::stats {

class NumpyGenerator {
 public:
  /** numpy.random.default_rng(seed) for a non-negative integer seed */
  explicit NumpyGenerator(std::uint64_t seed);
  /** the bit generator's next 64 bits (numpy `bit_generator.random_raw()`) */
  std::uint64_t nextRaw();
  /** numpy `Generator.random()`: a double in [0, 1) */
  double random();
 private:
  unsigned __int128 state_ = 0, increment_ = 0;
};

struct SignFlipConsistency {
  Eigen::VectorXd consistency;      // R1 [elements]
  Eigen::VectorXd p, familywiseP;   // [elements]
  double threshold = 0;             // the null maximum's 95th percentile
  double median = 0, medianNull95 = 0, medianP = 0;
  double neighbourCorrelation = 0;  // the mean correlation of neighbouring cells' unit fields
  std::int64_t cells = 0, groups = 0, patterns = 0;
};
/** Throws std::invalid_argument on no cell, components < 1 or not dividing the columns, patterns < 1 or cellsPerFlip < 1. */
SignFlipConsistency signFlipConsistency(const Eigen::MatrixXd& values, int components, std::int64_t patterns, NumpyGenerator& generator, std::int64_t cellsPerFlip = 1);

/** NaN with fewer than two whole groups. Throws std::invalid_argument on components < 1 or not dividing the columns, or cellsPerGroup < 1. */
double groupNeighbourCorrelation(const Eigen::MatrixXd& values, int components, std::int64_t cellsPerGroup);

struct DecorrelatedGrouping {
  std::int64_t cellsPerFlip = 1;      // the fewest consecutive cells whose group sums are practically uncorrelated (not reached: the largest tried)
  double correlation = 0;             // the neighbour correlation of the group sums at that grouping (NaN: fewer than two whole groups even one cell a group)
  bool reached = false;               // the correlation fell below the threshold
  std::vector<double> correlations;   // the correlation at each grouping tried (1, 2, ... cells)
};
/** Throws std::invalid_argument on components < 1 or not dividing the columns, or minimumGroups < 2. */
DecorrelatedGrouping fewestDecorrelatedCells(const Eigen::MatrixXd& values, int components, double threshold, std::int64_t minimumGroups);

}  // namespace nxr::stats
