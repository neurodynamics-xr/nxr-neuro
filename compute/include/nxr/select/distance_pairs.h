#pragma once
// nxr::select — PAIRS AT A DISTANCE (from lag.h). PURE MATHS (arrays in, arrays out).
//   pairsAtDistance(P, edges, d, …)  pairs of vertices at a distance d (± tolerance): straight-line, or along the graph's edges (Dijkstra, bounded); at most
//                                    `perVertex` a vertex, evenly spread (deterministic); their distances. The pairs a lag-against-distance is read at;
//                                    its summary is a measurement over distance bins (`measurement::reduce`), not a function here.
#include <Eigen/Dense>
#include <cstdint>

namespace nxr::select {

using Edges = Eigen::Matrix<int32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

enum class Metric { Straight, AlongEdges };
struct PairsAt {
  Edges pairs;                   // (u, v), u < v, each once
  Eigen::VectorXd distance;      // their distances
};
/** pairs at distance d ± tolerance·d; `edges` with `positions` give the graph for AlongEdges (edge lengths from the positions; ignored for Straight). */
PairsAt pairsAtDistance(const Eigen::MatrixXd& positions, const Edges& edges, double d, double tolerance, int perVertex, Metric metric);

}  // namespace nxr::select
