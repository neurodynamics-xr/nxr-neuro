#pragma once
// nxr::select — THE SPACE × TIME EXTENT OF AN EVENT on the joint cells (spec docs/specs/2026-11-13-cortical-events.md, phase 2b). PURE MATHS on matrices: a cell is (a spatial tile) × (a time tile);
// the functions never know where the matrices came from (a measurement of a source estimate, a synthetic array), never read a file, and never pick their inputs. Layout of every [S × T] matrix: row = spatial
// tile, column = time tile (Eigen's column-major storage makes a cell's node number frame × S + tile).
//
//   bitsFor / cellCode / cellOf    the Cortical Flow Studio atlas's joint code (tile << B) | frame, B = max(1, ceil(log2(max(2, F)))), and its inverse; a tile or a frame outside the layout is refused
//   spaceTimePairs                 the joint graph of a window of F time tiles: the level's touching tiles within a time tile, and each tile with itself in the next time tile (detectRegions runs on it unchanged)
//   standardize                    z = (amplitude − median_s) / SD_s per cell against its spatial tile's baseline; NaN where the tile has no spread (SD not > 0) or the cell has no value
//   windowRegions                  detectRegions on the window's cells, then the cell-count limit: the events of one burst, numbered by their lowest cell (earliest time tile, then lowest tile)
//   (an event's measures from its cells (count, union of tiles, area- and sample-weighted mean, largest peak, area by time tile) are MEASURES: `nxr::measurement::reduce` over the event's cells, `merge` over
//    its tiles: a reduction across cells is never a function here)
//   centres / pathLength           per event and time tile the vertex nearest the area-weighted centroid of its tiles' vertices (nxr::select::tileCentres), and the chord length of the centres' path (the
//                                  member's own measure `path_length`: a path, not a sum, so it is read from the centres here)
// A REDUCTION across cells is a measurement over the labelling's members (the labelling is the partition), never a loop inside a chain's node.
#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "nxr/select/regions.h"

namespace nxr::select {

using Pairs = std::vector<std::pair<std::int32_t, std::int32_t>>;
using Labels = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>;

/** max(1, ceil(log2(max(2, frames)))): the Studio atlas's bitsFor. std::invalid_argument for frames < 1. */
int bitsFor(std::int64_t frames);
/** (tile << bits) | frame. std::invalid_argument for a tile < 0, a frame outside 0 … frames − 1, or a code beyond 31 bits. */
std::int32_t cellCode(std::int64_t tile, std::int64_t frame, int bits, std::int64_t frames);
struct Cell { std::int32_t tile = 0, frame = 0; };
/** the inverse. std::invalid_argument for a negative code or a frame outside 0 … frames − 1 (a gap of the layout when frames is not a power of two). */
Cell cellOf(std::int32_t code, int bits, std::int64_t frames);

/** nodes frame × tiles + tile for frame 0 … frames − 1: (a, b) of `tilePairs` in every frame, and (tile, frame) – (tile, frame + 1). std::invalid_argument for a tile outside 0 … tiles − 1. */
Pairs spaceTimePairs(const Pairs& tilePairs, int tiles, int frames);

/** (amplitude(s, f) − median(s)) / sd(s) for a tile with sd > 0 (finite), NaN otherwise and for a NaN amplitude. `skipped` (optional) counts the tiles left out for no spread. */
Eigen::MatrixXd standardize(const Eigen::MatrixXd& amplitude, const Eigen::VectorXd& median, const Eigen::VectorXd& sd, int* skipped = nullptr);

struct WindowRegions {
  std::vector<std::int32_t> label;                    // [S × F] column-major (node = frame × S + tile): the event's ordinal within the window, −1 none
  std::vector<Region> regions;               // by ordinal; nodes are window nodes
};
/** `z` [S × F] standardized cells (NaN: no value), `tilePairs` of the level, `tileArea` [S], hysteresis enter ≥ exit on z, an event kept when it holds ≥ minCells cells and Σ tile areas of its cells ≥ minArea.
 *  std::invalid_argument as detectRegions, and for sizes that disagree or minCells < 1. */
WindowRegions windowRegions(const Eigen::MatrixXd& z, const Pairs& tilePairs, const Eigen::VectorXd& tileArea, double enter, double exit, int minCells = 1, double minArea = 0);

/** [E × T]: the vertex nearest the area-weighted centroid of the vertices of the event's tiles in that time tile (−1 where the event has no cell). `tileOfVertex` [V] (−1: none), `positions` [V × 3] row-major,
 *  `vertexArea` [V]. Only the time tiles in `frames` are visited (empty: all). */
Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic> centres(const Labels& label, int events, const std::vector<std::int32_t>& tileOfVertex, const std::vector<double>& positions,
                                                                   const std::vector<double>& vertexArea, const std::vector<int>& frames = {});
/** Σ of the straight distances between the centres of consecutive time tiles of each event (0 for an event of one time tile; NaN when a tile of its span has no centre). [E] */
Eigen::VectorXd pathLength(const Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>& centres, const std::vector<double>& positions);

}  // namespace nxr::select
