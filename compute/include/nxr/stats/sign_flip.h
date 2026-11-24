#pragma once
// nxr::stats — THE SIGN-FLIP (TIME-REVERSAL) TEST OF A DIRECTION'S CONSISTENCY over the cells of a time level, and the random draws it uses. PURE MATHS on arrays.
// Reference: nxr's flow maps (`nxr.flow.flowmap.flip_test`), the golden flow analysis's step 11: time reversal conjugates a cross-spectrum, so each cell's phase
// current changes sign; the cells' unit vectors are summed per element and their resultant length R1 compared with random sign patterns over the cells.
//
//   NumpyGenerator          numpy's `default_rng(seed)`: the SeedSequence of the seed, PCG64 (XSL-RR, 128-bit state), and `random()` = (next 64 bits >> 11) / 2^53.
//                           The same draws as numpy, bit for bit, so a statistic's p values reproduce a reference that drew with numpy (the golden's default_rng(0))
//   signFlipConsistency     values [cells x elements·components] (an element's components consecutive): U = each cell's vector cut to unit length per element (0 stays 0);
//                           R1_e = |Σ_cells U| / cells; `patterns` sign vectors over groups of `cellsPerFlip` consecutive cells, drawn as numpy's
//                           random((patterns, groups)) < 0.5 → −1 (else +1), in that order; per element p = (#(null R1 ≥ R1) + 1)/(patterns + 1); the family-wise
//                           p = (#(null max over elements ≥ R1) + 1)/(patterns + 1); the threshold = the null maximum's 95th percentile; the median R1 over elements,
//                           its null's 95th percentile and p (numpy's median and linear percentile). Also the mean correlation of neighbouring cells' unit fields
//                           (Σ_e U_s,e · U_s+1,e / elements, the mean over s): the check of whether single cells are exchangeable
#include <Eigen/Dense>
#include <cstdint>

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

}  // namespace nxr::stats
