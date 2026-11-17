// nxr::nearest — the NEAREST VERTEX of each query point: a k-d tree (Bentley 1975) over the vertices, written from the definition. PURE MATHS: arrays in, arrays out;
// no file, no subject, no knowledge of what the points are (fiber ends, a sphere's vertices, a cell's centre).
//
//   nearestVertices(positions [n x 3], query [q x 3])   the vertex nearest each query point (Euclidean; the lowest index on an exact tie) and its distance.
//   nearestVerticesBrute                                 the same answer by scanning every vertex: the reference the tree is tested against.
//
// Precision follows the input: float32 points give float32 squared distances (summed in the order x, y, z) and float32 distances; float64 gives float64. Both
// functions compute each squared distance by the same expression, so they agree on ties as well as on values.
#pragma once
#include <cstdint>
#include <vector>

namespace nxr::nearest {

template <class S>
struct Result {
  std::vector<std::int32_t> vertex;   // [q] the nearest vertex of each query point
  std::vector<S> distance;            // [q] its Euclidean distance, in the points' unit
};

/** Points are row-major [n x 3]. std::invalid_argument for a size that is not a multiple of 3, or for no vertex. */
template <class S>
Result<S> nearestVertices(const std::vector<S>& positions, const std::vector<S>& query);

template <class S>
Result<S> nearestVerticesBrute(const std::vector<S>& positions, const std::vector<S>& query);

}  // namespace nxr::nearest
