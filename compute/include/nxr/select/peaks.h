#pragma once
// nxr::select — POINT EVENTS ON A GRID: the candidates of a matched-filter detection and their suppression (spec docs/specs/2026-11-20-gabor-bursts.md, pieces 3 and 6; the reference
// detector's peaks and merge).
// PURE MATHS: a grid of values and the caller's levels in, positions out; points with their own resolution in, the kept ones out. The levels are the caller's (a statistic over a time partition,
// handed in as numbers): nothing here reduces over time.
//
//   localMaxima   the cells of a [rows x columns] grid that are at least every neighbour in their 3 x 3 box (the box clipped at the grid's edges: scipy maximum_filter, mode nearest) and at least
//                 their row's level; NaN is never a maximum and never beats one. Row-major order.
//   suppress      non-maximum suppression of point events within each event's own resolution: the strongest first; an event is dropped when a kept one of its group lies within the larger of the
//                 two time resolutions AND the larger of the two frequency resolutions (strict). The kept events' indices, strongest first (ties: the earlier index first).
#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace nxr::select {

std::vector<std::pair<Eigen::Index, Eigen::Index>> localMaxima(const Eigen::Ref<const Eigen::MatrixXd>& values, const Eigen::Ref<const Eigen::VectorXd>& level);

/** a point event: its place in time (s) and frequency (Hz), its resolution in each, its strength, its group (events of different groups never suppress each other: a channel) */
struct PointEvent {
  double t = 0, f = 0, tResolution = 0, fResolution = 0, strength = 0;
  std::int64_t group = 0;
};
std::vector<std::size_t> suppress(const std::vector<PointEvent>& events);

}  // namespace nxr::select
