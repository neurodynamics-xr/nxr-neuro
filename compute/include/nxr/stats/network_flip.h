#pragma once
// nxr::stats — THE NULLS OF THE FLOW THROUGH A NETWORK, ported exactly from nxr's flow through the fibre network (`flownet.py`), with numpy's draws bit for bit. PURE MATHS on arrays.
//
//   networkSignFlip         the reference's `flip`: values [segments x elements] (each segment's edge flow); observed = the sum over segments; `patterns` sign vectors drawn as numpy's
//                           random((patterns, segments)) < 0.5 → −1 (else +1); null = signs · values; per element p = (#(|null| >= |observed|) + 1) / (patterns + 1); the family-wise p
//                           over elements on the sums normalised by Σ |values| (floor 1e-30): p = (#(null max over elements of |null| / Σ|values| >= |observed| / Σ|values|) + 1) / (patterns + 1)
//   numpyChoice             numpy `Generator.choice(population, size, replace=False)` for a population of at most 10 000 (Floyd's algorithm, then the shuffle): the same indices in the same
//                           order, drawing the same numbers from the generator
//   digitize                numpy `digitize(x, edges)` for increasing edges (right = False): the count of edges <= x
//   distanceMatchedControl  the reference's control: for each distance bin the joined elements occupy (ascending), as many not-joined elements of that bin as joined ones (or all of them),
//                           drawn by numpyChoice from the not-joined elements of the bin in index order
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

#include "nxr/stats/sign_flip.h"

namespace nxr::stats {

struct NetworkSignFlip {
  Eigen::VectorXd sum;           // [elements] the observed sum over segments
  Eigen::VectorXd p, familywiseP;
  std::int64_t segments = 0, patterns = 0;
};
/** Throws std::invalid_argument on no segment or patterns < 1. */
NetworkSignFlip networkSignFlip(const Eigen::MatrixXd& segmentValues, std::int64_t patterns, NumpyGenerator& generator);

/** population <= 10 000, 0 <= size <= population. */
std::vector<std::int64_t> numpyChoice(std::int64_t population, std::int64_t size, NumpyGenerator& generator);

std::vector<std::int64_t> digitize(const Eigen::VectorXd& x, const std::vector<double>& edges);

std::vector<std::uint8_t> distanceMatchedControl(const std::vector<std::uint8_t>& joined, const std::vector<std::int64_t>& bin, NumpyGenerator& generator);

}  // namespace nxr::stats
