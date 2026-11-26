// nxr::select — THE LEVELS OF ONE MANIFOLD: addressing (level, code) and navigation at every level of a stored grid (spec docs/specs/2026-11-26-levels-of-one-manifold.md).
//
// A level is not a manifold row: it is COMPUTED from the stored manifold (its fine complex: vertices with ⋆0, oriented edges d0 with ⋆1, faces d1) and the grid's
// partition of its vertices, and it is a cell complex in its own right, so navigation is the same at every level:
//   levelComplex      the QUOTIENT of a fine complex by a vertex partition: cells, edges (tail < tip, pair order), d0, ⋆0 (summed), ⋆1 (GALERKIN: the summed fine ⋆1 of the
//                     fine edges crossing a boundary, so d0ᵀ⋆1 d0 at the level = Pᵀ L P), the CROSSING map E (d0_fine P = E d0_level), the JUNCTION faces (a fine face whose
//                     three corners lie in three cells; d1_fine E = J d1_level), and the halfedges (twin, next) where the level is a valid surface complex, else refused with why
//   vertexLevel       the fine complex itself (each vertex its own cell), with the grid's codes for its parents
//   dyadicLevel       a level l of a dyadic grid (the codes at its depth: level l's cell of a vertex is code >> (depth - l), selection.h's closed form)
//   lineLevel         a level k of a time line (tiles of 2^k samples from the first; closed form, nothing per sample is held)
//   neighbours, outgoing, twin, next, parent, children, area, laplacian                           navigation on one level
//   cellMap, reductionOperator, expansionOperator, reduce                                         between two levels of one ladder, per merge rule (reduce is roll_values)
//   productLevel, jointCode, jointAddress, productNeighbours, productOperators                    (a surface level) x (a time level), separable; the dispersion coupling is a typed slot
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace nxr::select {

using SparseMatrix = Eigen::SparseMatrix<double>;   // column-major (CSC)

/** the level of a surface's own vertices: the manifold as meshed, below every dyadic level (a time line's samples are its level 0) */
constexpr int kVertexLevel = -2;
/** no level (the parent of a coarsest cell, the children of a sample) */
constexpr int kNoLevel = -3;

/** the fine complex a level coarsens: its operators as DEC gives them (nxr::operators::assembleDECOperators: the operator mesh; embeddedDEC: the input) */
struct FineComplex {
  SparseMatrix d0;                       // [E x V]: -1 at an edge's tail, +1 at its tip
  SparseMatrix d1;                       // [F x E] (optional, 0 x 0: no faces): the signed edges of each face, in its orientation
  Eigen::VectorXd star0;                 // [V] the vertices' dual areas (the caller's measure: barycentric, mixed Voronoi)
  Eigen::VectorXd star1;                 // [E] the edges' conductances (the cotan weights; raw, never floored here)
};

/** a cell's address on one ladder: its level (its stored label) and its code (the grid's member number at that level) */
struct LevelAddress {
  int level = kNoLevel;
  std::int64_t code = -1;
};
/** a cell's children: one level finer, their codes (empty: none) */
struct Children {
  int level = kNoLevel;
  std::vector<std::int64_t> codes;
};

struct Level {
  enum class Ladder { vertices, dyadic, time };
  Ladder ladder = Ladder::vertices;
  int level = kVertexLevel;              // dyadic: l (2^l tiles a piece; 0 the coarsest); time: k (tiles of 2^k samples; 0 the samples); vertices: kVertexLevel
  int depth = 0;                         // dyadic (and the vertices of a dyadic grid): the grid's finest level
  std::int64_t slots = 0;                // dyadic: the piece slots its codes reserve (level l has slots << l cells)
  std::vector<std::int32_t> codes;       // dyadic and vertices: the grid's codes at its depth, one per vertex (-1: none)
  std::int64_t samples = 0;              // time: the line's samples
  double rate = 0;                       // time: samples a second
  std::int64_t cells = 0;
  std::vector<std::int32_t> cellOf;      // [fine vertices] each vertex's cell (-1: none); empty for a time level (closed form)
  Eigen::Matrix<std::int32_t, Eigen::Dynamic, 2, Eigen::RowMajor> edges;   // [E x 2] tail < tip, in pair order
  SparseMatrix d0;                       // [E x cells]
  Eigen::VectorXd star0, star1;          // [cells], [E]
  SparseMatrix crossing;                 // [fine edges x E] the signed crossing map (empty for a time level: one fine edge per boundary, `crossingEdge`)
  Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor> faces;   // [F x 3] the junctions' cells, in the fine face's orientation
  std::vector<std::int64_t> junctions;   // [F] the fine face of each level face
  SparseMatrix d1;                       // [F x E]
  bool valid = false;                    // a valid surface complex: halfedges (twin, next) exist
  std::string why;                       // when not valid: why (the edge, the cell, the Euler characteristic)
  std::vector<std::int32_t> heTwin;      // [3F] halfedge 3f + i goes faces(f, i) -> faces(f, (i + 1) % 3); its twin (-1 on the boundary); empty when not valid
  std::vector<std::int32_t> heEdge;      // [3F] its edge
  std::vector<std::int64_t> adjacencyOffsets, adjacencyCells, adjacencyEdges;   // CSR: each cell's neighbours and the edge to each
  std::int64_t negativeConductances = 0; // level edges whose Galerkin ⋆1 is negative (from negative fine cotan weights, an open rim)
  std::int64_t fineEuler = 0;            // V - E + F of the fine complex (the vertices in a cell)
  std::int64_t euler = 0;                // V - E + F over the non-empty cells
};

/** THE QUOTIENT of a fine complex by a partition of its vertices (`cellOf`, -1: none) into `cells` cells. std::invalid_argument for sizes that disagree, a cell outside
 *  0 .. cells - 1, or a d0 row that is not one -1 and one +1. The ladder fields (level, depth, slots, codes) are the caller's: `dyadicLevel`, `vertexLevel`. */
Level levelComplex(const FineComplex& fine, const std::vector<std::int32_t>& cellOf, std::int64_t cells);
/** the fine complex as a level (each vertex its own cell), with a dyadic grid's codes (h 2^depth + t) for its parents (codes may be empty: no grid) */
Level vertexLevel(const FineComplex& fine, const std::vector<std::int32_t>& codes = {}, int depth = 0, std::int64_t slots = 0);
/** level l (0 .. depth) of a dyadic grid whose codes at `depth` are `codes` (h 2^depth + t, -1: none; selection.h's closed form: the cell is code >> (depth - l)) */
Level dyadicLevel(const FineComplex& fine, const std::vector<std::int32_t>& codes, int depth, std::int64_t slots, int level);
/** level k of a time line of `samples` samples at `rate` Hz: tiles [m 2^k, (m + 1) 2^k) from the first sample, the last as long as the line; ⋆0 a tile's duration (samples / rate),
 *  ⋆1 = rate (Galerkin: one fine edge, of conductance 1 / dt, crosses each boundary). Closed form. std::invalid_argument for samples < 1, rate <= 0, k outside 0 .. 62. */
Level lineLevel(std::int64_t samples, double rate, int k);

// ── on one level ─────────────────────────────────────────────────────────────────────────
/** a cell's neighbours (the cells an edge joins it to), ascending */
std::vector<std::int64_t> neighbours(const Level& level, std::int64_t cell);
/** a cell's edges: their indices (into `edges`), in the order of `neighbours` */
std::vector<std::int64_t> incidentEdges(const Level& level, std::int64_t cell);
/** a cell's outgoing halfedges (3f + i with faces(f, i) == cell); std::invalid_argument when the level is not valid (the message is `why`) */
std::vector<std::int64_t> outgoing(const Level& level, std::int64_t cell);
/** a halfedge's twin (-1: on the boundary) and next (within its face); std::invalid_argument when the level is not valid (the message is `why`) */
std::int64_t twin(const Level& level, std::int64_t halfedge);
std::int64_t next(const Level& level, std::int64_t halfedge);
/** a halfedge's tail and tip cells */
LevelAddress tail(const Level& level, std::int64_t halfedge);
LevelAddress tip(const Level& level, std::int64_t halfedge);
/** the cell one level coarser holding `cell` (level kNoLevel: none, the coarsest) */
LevelAddress parent(const Level& level, std::int64_t cell);
/** the cells one level finer that `cell` holds (dyadic finest level: the vertices; a vertex, a sample: none) */
Children children(const Level& level, std::int64_t cell);
/** a cell's ⋆0: its area (a surface) or its duration (a line) */
double area(const Level& level, std::int64_t cell);
/** the time level's fine edge crossing level edge e (between tiles e and e + 1): sample 2^k (e + 1) - 1 to 2^k (e + 1) */
std::int64_t crossingEdge(const Level& level, std::int64_t edge);
/** d0ᵀ ⋆1 d0 [cells x cells] */
SparseMatrix laplacian(const Level& level);

// ── between two levels of one ladder ─────────────────────────────────────────────────────
/** each cell of `from` its cell in `to` (a coarser level of the same grid or line: the vertices to a dyadic level, a dyadic level to a coarser one, a time level to a coarser
 *  one); std::invalid_argument for levels of different ladders or `to` not coarser */
std::vector<std::int64_t> cellMap(const Level& from, const Level& to);
/** the reduction [to.cells x from.cells] of a 0-form by merge rule `sum` (Pᵀ) or `mean` (⋆0_to⁻¹ Pᵀ ⋆0_from; an empty cell's row is zero); `max`, `min` have no matrix (`reduce`),
 *  `none` and `derived` are never reduced: std::invalid_argument naming the rule */
SparseMatrix reductionOperator(const Level& from, const Level& to, const std::string& rule);
/** the expansion [fine.cells x coarse.cells] back: `mean` piecewise constant (P), `sum` the coarse value split by ⋆0 share (⋆0_fine P ⋆0_coarse⁻¹); an order statistic and `none`
 *  have none: std::invalid_argument */
SparseMatrix expansionOperator(const Level& coarse, const Level& fine, const std::string& rule);
/** values [rows x from.cells] reduced to `to` by `rule` (sum, mean weighted by ⋆0_from, max, min) through nxr::measurement::roll_values (the one roll-up) */
Eigen::MatrixXd reduce(const Level& from, const Level& to, const std::string& rule, const Eigen::MatrixXd& values);

// ── the product (a surface level) x (a time level) ───────────────────────────────────────
/** how the product's levels are tied: SEPARABLE (chosen independently) or by a DISPERSION relation (D_l / dt_k = c, 2 pi f_k = c sqrt(lambda_l)): a typed slot, refused in this step */
struct Coupling {
  enum class Kind { separable, dispersion };
  Kind kind = Kind::separable;
  double speed = std::numeric_limits<double>::quiet_NaN();   // m/s (dispersion)
  int sense = 1;                                             // +1 | -1 (dispersion)
};
struct ProductLevel {
  Level space, time;
  Coupling coupling;
  std::int64_t cells() const { return space.cells * time.cells; }
};
struct JointAddress {
  LevelAddress space, time;
};
/** the product of a surface level (vertices or dyadic) and a time level; std::invalid_argument for other ladders and for a dispersion coupling (not built yet) */
ProductLevel productLevel(Level space, Level time, Coupling coupling = {});
/** the joint code: selection::product_code({space cells, time cells}, {space, time}), space fastest (a [space x time] column-major array's index) */
std::int64_t jointCode(const ProductLevel& p, std::int64_t space, std::int64_t time);
JointAddress jointAddress(const ProductLevel& p, std::int64_t code);
/** the space neighbours in the same time tile, then the same space cell in the previous and next time tile (spaceTimePairs's graph), ascending */
std::vector<std::int64_t> productNeighbours(const ProductLevel& p, std::int64_t code);
/** the product's operators: d0 = [d0_s ⊗ I_t ; I_s ⊗ d0_t] (space edges x time cells first, then space cells x time edges), ⋆0 = ⋆0_s ⊗ ⋆0_t, ⋆1 = [⋆1_s ⊗ ⋆0_t ; ⋆0_s ⊗ ⋆1_t] */
struct ProductOperators {
  SparseMatrix d0;
  Eigen::VectorXd star0, star1;
};
ProductOperators productOperators(const ProductLevel& p);

}  // namespace nxr::select
